/*
 * thermal_streamer.ino
 * ---------------------------------------------------------------------------
 * Streams raw frames from an MLX90640 thermal camera over WiFi to a PC.
 *
 * HARDWARE
 *   Board   : Arduino UNO R4 WiFi (any board using the WiFiS3 library)
 *   Sensor  : Melexis MLX90640, 32x24 far-infrared array, I2C at 400 kHz
 *   Wiring  : SDA/SCL to the board's I2C pins, 3V3 and GND.
 *             The sensor is 3.3 V -- do not feed it 5 V.
 *
 * DEPENDENCIES
 *   MLX90640_API.h / MLX9064X_I2C_Driver.h from the Melexis driver
 *   (as redistributed by SparkFun). These are .h/.cpp files dropped into
 *   the sketch folder, not a Library Manager entry.
 *
 * WHAT THIS SKETCH DOES NOT DO
 *   No temperature conversion happens on the board. The Arduino ships the
 *   sensor's raw ADC words plus its factory calibration EEPROM, and the PC
 *   does the (comparatively expensive) maths to turn those into degrees.
 *   This keeps the microcontroller's job to "read I2C, push bytes".
 *
 * WIRE PROTOCOL (Arduino -> PC, TCP, no framing beyond what is listed here)
 *
 *   Each message is a 2-byte ASCII tag followed by a fixed-size payload of
 *   little-endian uint16_t values. There is no length prefix, no checksum,
 *   and no delimiter -- the receiver must know the sizes in advance.
 *
 *     "EE" + 832 uint16  (1664 bytes)  sent once, right after connecting.
 *                                      The sensor's calibration EEPROM dump.
 *                                      The PC needs this before it can
 *                                      convert any frame to temperatures.
 *
 *     "FR" + 834 uint16  (1668 bytes)  sent continuously, two per full image.
 *                                      A raw subpage frame (see loop()).
 *
 *   Little-endian is not a choice made here -- it is simply how the ARM
 *   Cortex-M4 in this board lays out a uint16_t in memory, and the bytes are
 *   handed to client.write() straight from that memory. A PC receiver in
 *   Python would unpack these with '<834H'.
 * ---------------------------------------------------------------------------
 */

// ===========================================================================
// WiFi
// ===========================================================================

#include <WiFiS3.h>

// Credentials live in a separate header so this file can be shared safely.
// See arduino_secrets.h.
#include "arduino_secrets.h"

char ssid[] = SECRET_SSID;
char pass[] = SECRET_PASS;

// Radio state, updated by WiFi.begin(). Starts "idle" meaning: powered on,
// not associated with any access point yet.
int status = WL_IDLE_STATUS;

const char* host = SECRET_HOST;  // PC running the receiving script
const int   port = SECRET_PORT;

WiFiClient client;

// ===========================================================================
// MLX90640 thermal camera
// ===========================================================================

#include <Wire.h>

// Selects the 90640 code paths in the driver headers below. The same driver
// also supports the MLX90641 (16x12); leaving this undefined would change
// which branches compile.
#define USE_MLX90640

#include "MLX90640_API.h"
#include "MLX9064X_I2C_Driver.h"

// Factory-default 7-bit I2C address. "Unshifted" means Wire will add the
// read/write bit itself -- do not pre-shift this to 0x66.
const byte MLX90640_address = 0x33;

/*
 * Frame buffer layout (834 words):
 *
 *   [0   .. 767]  768 pixel ADC words, 32 columns x 24 rows, row-major:
 *                     pixel(row, col) = MLX90640Frame[row * 32 + col]
 *                 On any one read only half of these are fresh -- see loop().
 *
 *   [768 .. 831]  64 auxiliary/housekeeping words: VDD sensor reading,
 *                 ambient temperature data, gain, and the shielded
 *                 compensation pixels. The sensor drifts with its own
 *                 temperature and supply voltage, so every frame carries
 *                 these and the PC's conversion maths uses them to correct
 *                 that frame. This is why calibration cannot simply be
 *                 applied once and forgotten.
 *
 *   [832]         control register 1 (sensor register 0x800D)
 *   [833]         subpage number, already masked by the driver to a bare
 *                 0 or 1 -- not a raw register value.
 *
 * Words 0..831 are one contiguous I2C read from sensor RAM 0x0400 (pixels)
 * straight through 0x0700 (auxiliary). The driver then appends 832 and 833
 * with two small extra reads, which is why the buffer is 834 rather than 832.
 *
 * The EEPROM dump is also 832 words -- a different, unrelated size. They are
 * easy to confuse, which is why both are named constants below.
 */
static const uint16_t MLX90640_FRAME_WORDS  = 834;
static const uint16_t MLX90640_EEPROM_WORDS = 832;

static uint16_t MLX90640Frame[MLX90640_FRAME_WORDS];
static paramsMLX90640 params;  // calibration coefficients, unpacked from EEPROM

// Guards the one-shot calibration transmission so a future reconnect does not
// silently resend it and desynchronise the PC's parser.
bool ee_sent = false;

// Refresh rate code passed to MLX90640_SetRefreshRate(). The register takes a
// 3-bit exponent, not a rate in Hz:
//   0x00=0.5Hz  0x01=1Hz  0x02=2Hz  0x03=4Hz  0x04=8Hz  0x05=16Hz ...
// Note this is the SUBPAGE rate, so 4 Hz here yields 2 complete images per
// second. Pushing higher costs more I2C bandwidth and raises sensor noise.
static const uint8_t MLX90640_REFRESH_4HZ = 0x03;


// ===========================================================================
// Setup
// ===========================================================================

void setup() {

    // 115200 baud matches the rate the Serial Monitor should be set to.
    Serial.begin(115200);

    Wire.begin();
    Wire.setClock(400000);  // 400 kHz "fast mode"; the default 100 kHz is too
                            // slow to pull 834 words at the chosen frame rate
    delay(100);             // let the sensor finish its power-on sequence

    // ----- Bring up the radio ---------------------------------------------

    // Distinguishes "no WiFi hardware / firmware not responding" from the
    // ordinary "cannot find the network" case handled further down.
    if (WiFi.status() == WL_NO_MODULE) {
        Serial.println("Communication with WiFi module failed!");
        while (true);  // unrecoverable -- halt rather than spin uselessly
    }

    // String comparison against the version this library was built for.
    // Informational only; an older module usually still works.
    String fv = WiFi.firmwareVersion();
    if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
        Serial.println("Please upgrade the firmware");
    }

    // Retry until associated. WiFi.begin() on this board blocks for a while
    // on its own; the 10 s delay on top gives a slow AP room to respond and
    // stops the serial output from scrolling past unreadably.
    while (status != WL_CONNECTED) {
        Serial.print("Attempting to connect to WPA SSID: ");
        Serial.println(ssid);

        status = WiFi.begin(ssid, pass);

        Serial.print("WiFi.status() = ");
        Serial.println(status);

        delay(10000);
    }

    Serial.println("\nWiFi connected");
    printCurrentNet();
    printWifiData();

    // ----- Open the TCP session to the PC ---------------------------------

    if (client.connect(host, port)) {
        Serial.println("Connected to server (PC)");

        // Drain and echo whatever greeting the PC sends, if any. Purely a
        // debugging aid -- nothing in the protocol requires a handshake.
        // Waits up to 5 s for the first byte, then prints what has arrived.
        unsigned long timeout = millis();
        while (!client.available() && millis() - timeout < 5000) {}

        while (client.available()) {
            char c = client.read();
            Serial.print(c);
        }

    } else {
        // NOTE: execution continues either way. If this fails, the sketch
        // will still talk to the sensor but every client.write() below is a
        // no-op, so nothing reaches the PC and nothing retries. Worth adding
        // a reconnect path if you want this to survive the PC restarting.
        Serial.println("Connection to PC failed");
    }

    // ----- Bring up the sensor --------------------------------------------

    if (isConnected() == false) {
        Serial.println("MLX9064x not detected at default I2C address. Please check wiring. Freezing.");
        while (1);
    }

    // Renamed from 'status' to avoid shadowing the global WiFi status above.
    // The original compiled fine, but two different meanings behind one name
    // in overlapping scopes is a trap.
    int mlxStatus;

    // Each sensor is individually calibrated at the factory; its coefficients
    // live in on-chip EEPROM and must be read out before any conversion.
    // Declared locally because the PC, not the Arduino, does the converting --
    // once these bytes are on the wire the board has no further use for them.
    uint16_t eeMLX90640[MLX90640_EEPROM_WORDS];

    mlxStatus = MLX90640_DumpEE(MLX90640_address, eeMLX90640);
    if (mlxStatus != 0) {
        Serial.println("Failed to load system parameters");
    }

    // Unpacks the raw EEPROM into the struct the conversion API expects.
    // Kept here even though the board never converts anything, because it
    // also validates the dump -- a failure here means the bytes about to be
    // sent to the PC are suspect.
    mlxStatus = MLX90640_ExtractParameters(eeMLX90640, &params);
    if (mlxStatus != 0) {
        Serial.println("Parameter extraction failed");
    }

    MLX90640_SetRefreshRate(MLX90640_address, MLX90640_REFRESH_4HZ);

    // Ship the calibration to the PC exactly once, before any frames, so the
    // receiver has everything it needs to interpret what follows.
    if (client && !ee_sent) {
        client.write((const uint8_t*)"EE", 2);
        client.write((uint8_t*)eeMLX90640, MLX90640_EEPROM_WORDS * 2);
        ee_sent = true;
    }
}


// ===========================================================================
// Main loop
// ===========================================================================

void loop() {
    int mlxStatus;

    /*
     * The MLX90640 never refreshes a whole image at once. It updates a
     * chess-board pattern of pixels, alternating between two "subpages":
     *
     *   subpage 0 -- the squares where (row + column) is even
     *   subpage 1 -- the squares where (row + column) is odd
     *
     * A single read therefore returns a complete 768-word buffer in which
     * only half the values are current; the rest are left over from the
     * previous read. Both subpages are needed before the PC can assemble a
     * fully up-to-date 32x24 image, which is why this runs twice.
     *
     * MLX90640_GetFrameData() blocks until the sensor flags a new frame as
     * ready, so the pacing of this loop is set by the refresh rate configured
     * in setup(), not by any delay here.
     */
    for (byte subpage = 0; subpage < 2; subpage++) {

        mlxStatus = MLX90640_GetFrameData(MLX90640_address, MLX90640Frame);

        // Word 833 is the subpage number, already reduced by the driver to a
        // bare 0 or 1. On a healthy bus this prints a clean alternating
        // 0, 1, 0, 1 -- if it sticks on one value you are only ever receiving
        // half the image. (mlxStatus returns the same subpage number, or a
        // negative value on an I2C read error; it is currently ignored, so a
        // corrupted read still gets shipped to the PC as though it were good.)
        Serial.print("[833]=");
        Serial.print(MLX90640Frame[833]);

        if (client) {
            client.write((const uint8_t*)"FR", 2);
            client.write((uint8_t*)MLX90640Frame, MLX90640_FRAME_WORDS * 2);
        }

        // Blocks until the outgoing buffer is drained, trading throughput for
        // the guarantee that frames do not pile up in RAM. Sits outside the
        // `if (client)` guard, matching the original -- harmless on a dead
        // client, but it is doing nothing useful there either.
        client.flush();
    }
}


// ===========================================================================
// Diagnostics helpers
// ===========================================================================

// Prints a 6-byte MAC as colon-separated hex, zero-padding single digits
// (Serial.print(0x0A, HEX) would otherwise emit "A" rather than "0A").
void printMacAddress(byte mac[]) {
    for (int i = 0; i < 6; i++) {
        if (mac[i] < 16) Serial.print("0");
        Serial.print(mac[i], HEX);
        if (i < 5) Serial.print(":");
    }
    Serial.println();
}

// This board's own addressing. The IP is what you would point a PC-side tool
// at if you ever reversed the direction of the connection.
void printWifiData() {
    IPAddress ip = WiFi.localIP();
    Serial.print("IP Address: ");
    Serial.println(ip);

    byte mac[6];
    WiFi.macAddress(mac);
    Serial.print("MAC address: ");
    printMacAddress(mac);
}

// Details of the access point currently associated with. RSSI is the useful
// one for field work: roughly -50 dBm is excellent, -70 is marginal, and
// below -80 expect dropped frames.
void printCurrentNet() {
    Serial.print("SSID: ");
    Serial.println(WiFi.SSID());

    // BSSID is the router's own MAC -- distinguishes between multiple access
    // points broadcasting the same SSID.
    byte bssid[6];
    WiFi.BSSID(bssid);
    Serial.print("BSSID: ");
    printMacAddress(bssid);

    long rssi = WiFi.RSSI();
    Serial.print("signal strength (RSSI):");
    Serial.println(rssi);

    byte encryption = WiFi.encryptionType();
    Serial.print("Encryption Type:");
    Serial.println(encryption, HEX);
    Serial.println();
}

// Probes the I2C bus by starting a transmission and checking for an ACK.
// endTransmission() returns 0 only if a device acknowledged that address,
// so this detects a missing or miswired sensor without reading any registers.
boolean isConnected() {
#ifndef USE_MLX90641
    Wire.beginTransmission((uint8_t)MLX90640_address);
#else
    Wire.beginTransmission((uint8_t)MLX90641_address);
#endif
    if (Wire.endTransmission() != 0) {
        return (false);  // no ACK -- nothing at this address
    }
    return (true);
}
