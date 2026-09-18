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
 *     "EE" + 832 uint16  (1664 bytes)  sent once per boot, on the first
 *                                      successful connection, before any
 *                                      frames. The sensor's calibration
 *                                      EEPROM dump, needed by the PC before
 *                                      it can convert any frame to
 *                                      temperatures. It is NOT repeated on a
 *                                      reconnect, so a PC-side script that
 *                                      restarts mid-run must be restarted
 *                                      alongside the board, or it will
 *                                      receive frames it cannot convert.
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

/*
 * Connection state.
 *
 * Calibration is sent once per boot, not per connection. The dump is burned
 * into the sensor at the factory and never changes, so re-sending it on a
 * reconnect would only duplicate bytes the PC already holds.
 *
 * The flag latches on the first successful send rather than simply being set
 * in setup(), so that a board powered on before the PC-side script is running
 * still delivers calibration once the first connection succeeds.
 */
bool ee_sent = false;

// Timestamp of the last reconnect attempt, for the non-blocking retry in
// loop(). Unsigned subtraction makes the comparison safe across the ~49-day
// millis() rollover, so no special case is needed for it.
unsigned long lastReconnectAttempt = 0;
static const unsigned long RECONNECT_INTERVAL_MS = 5000;

// Diagnostics: frames rejected since boot, printed alongside the subpage
// trace so a bus that is quietly degrading is visible in the serial output.
unsigned long badFrames = 0;

// Refresh rate code passed to MLX90640_SetRefreshRate(). The register takes a
// 3-bit exponent, not a rate in Hz:
//   0x00=0.5Hz  0x01=1Hz  0x02=2Hz  0x03=4Hz  0x04=8Hz  0x05=16Hz ...
// Note this is the SUBPAGE rate, so 4 Hz here yields 2 complete images per
// second. Pushing higher costs more I2C bandwidth and raises sensor noise.
static const uint8_t MLX90640_REFRESH_4HZ = 0x03;


// Forward declarations. The Arduino IDE generates these automatically, but
// stating them makes the file valid plain C++ too -- useful if you ever
// compile it outside the IDE.
bool connectToServer();
bool sendCalibration();
boolean isConnected();
void printCurrentNet();
void printWifiData();
void printMacAddress(byte mac[]);


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

    // ----- Bring up the sensor --------------------------------------------
    //
    // Moved ahead of the TCP connection (it came after, originally) because
    // connectToServer() now sends the calibration dump as part of opening a
    // session, and that requires a working sensor to read the dump from.

    if (isConnected() == false) {
        Serial.println("MLX9064x not detected at default I2C address. Please check wiring. Freezing.");
        while (1);
    }

    MLX90640_SetRefreshRate(MLX90640_address, MLX90640_REFRESH_4HZ);

    // ----- Open the TCP session to the PC ---------------------------------
    //
    // A failure here is no longer terminal: loop() retries on its own, so the
    // board can be powered on before the PC-side script is running.
    connectToServer();
}


// ===========================================================================
// Connection handling
// ===========================================================================

/*
 * Reads the sensor's factory calibration EEPROM and forwards it to the PC.
 *
 * Called once per boot. The dump is read
 * fresh from the sensor each time instead of being cached in a global: at
 * 1664 bytes that would be a permanent bite out of RAM, which is already at
 * roughly two-thirds occupancy on this board. Here it lives on the stack for
 * the few milliseconds the function runs and then goes away.
 *
 * Returns true only if the dump was read, validated and sent in full.
 */
bool sendCalibration() {
    uint16_t eeMLX90640[MLX90640_EEPROM_WORDS];

    int mlxStatus = MLX90640_DumpEE(MLX90640_address, eeMLX90640);
    if (mlxStatus != 0) {
        Serial.println("Failed to load system parameters");
        return false;
    }

    // Unpacks the raw EEPROM into the struct the conversion API expects.
    // The board never converts anything itself, but this doubles as a
    // validity check: if extraction fails, the bytes are not worth sending.
    mlxStatus = MLX90640_ExtractParameters(eeMLX90640, &params);
    if (mlxStatus != 0) {
        Serial.println("Parameter extraction failed -- not sending calibration");
        return false;
    }

    if (!client.connected()) {
        return false;
    }

    client.write((const uint8_t*)"EE", 2);
    client.write((uint8_t*)eeMLX90640, MLX90640_EEPROM_WORDS * 2);
    client.flush();

    Serial.println("Calibration sent");
    return true;
}

/*
 * Opens a session to the PC, sending calibration if it has not gone out yet.
 *
 * Returns true once the link is up and the PC has calibration -- on a
 * reconnect that means simply that the socket opened, since the dump was
 * already delivered earlier in this boot.
 */
bool connectToServer() {
    // Releases any half-open socket from a previous session. Without this the
    // client can hold onto a stale descriptor and refuse to reconnect.
    client.stop();

    Serial.print("Connecting to ");
    Serial.print(host);
    Serial.print(":");
    Serial.println(port);

    if (!client.connect(host, port)) {
        Serial.println("Connection to PC failed");
        return false;
    }

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

    // First successful connection of this boot carries the calibration dump;
    // later reconnects skip it, since the PC already has it.
    if (!ee_sent) {
        ee_sent = sendCalibration();
        return ee_sent;
    }

    return true;
}


// ===========================================================================
// Main loop
// ===========================================================================

void loop() {

    /*
     * ----- Connection maintenance -----------------------------------------
     *
     * Checked before any sensor work, so a dead socket is dealt with rather
     * than having frames read and silently discarded into a closed client.
     *
     * client.connected() is true while the session is usable -- including
     * when the peer has closed but unread data is still buffered -- and goes
     * false once the connection is genuinely gone. That covers the PC-side
     * script being restarted, which is the common case.
     *
     * The retry is non-blocking: if it is not yet time to try again the
     * function simply returns and the next pass through loop() re-checks.
     * A blocking delay here would be worse than useless, since client.connect()
     * already stalls for its own timeout on a host that is not listening.
     */
    if (!client.connected()) {
        if (millis() - lastReconnectAttempt < RECONNECT_INTERVAL_MS) {
            return;
        }
        lastReconnectAttempt = millis();

        Serial.println("Link down -- reconnecting");
        connectToServer();
        return;  // stream on the next pass, once the link is settled
    }

    // Recovery path only: the link is up but calibration never made it out,
    // because an earlier attempt hit an I2C error. Retries until it succeeds,
    // since frames are meaningless to the PC without that dump. Once the flag
    // latches this never runs again for the rest of the boot.
    if (!ee_sent) {
        ee_sent = sendCalibration();
        return;
    }

    /*
     * ----- Frame acquisition ----------------------------------------------
     *
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
    for (byte i = 0; i < 2; i++) {

        int mlxStatus = MLX90640_GetFrameData(MLX90640_address, MLX90640Frame);

        /*
         * Validation, in increasing order of subtlety. Anything that fails is
         * dropped rather than transmitted: a dropped frame costs the PC one
         * stale half-image for a fraction of a second, whereas a corrupt one
         * feeds garbage into the temperature conversion and shows up as
         * plausible-looking but wrong readings.
         */

        // 1. Driver-level failure. On success this API returns the subpage
        //    number; anything negative is an I2C or timeout error.
        if (mlxStatus < 0) {
            badFrames++;
            Serial.print(" [read error ");
            Serial.print(mlxStatus);
            Serial.print("]");
            continue;
        }

        uint16_t subpage = MLX90640Frame[833];

        // 2. Out-of-range subpage. The driver masks this to 0 or 1, so any
        //    other value means the buffer is not holding what we think it is
        //    -- a partial read, or a write that landed at the wrong offset.
        if (subpage > 1) {
            badFrames++;
            Serial.print(" [bad subpage ");
            Serial.print(subpage);
            Serial.print("]");
            continue;
        }

        // 3. Disagreement between the two places the subpage is reported.
        //    Both derive from the same read, so they should always match;
        //    if they do not, the buffer was disturbed mid-transfer.
        if ((int)subpage != mlxStatus) {
            badFrames++;
            Serial.print(" [subpage mismatch ");
            Serial.print(mlxStatus);
            Serial.print("/");
            Serial.print(subpage);
            Serial.print("]");
            continue;
        }

        // Trace: healthy output alternates 0, 1, 0, 1. A value that sticks
        // means only half the image is ever refreshing.
        Serial.print("[833]=");
        Serial.print(subpage);

        client.write((const uint8_t*)"FR", 2);
        client.write((uint8_t*)MLX90640Frame, MLX90640_FRAME_WORDS * 2);

        // Blocks until the outgoing buffer is drained, trading throughput for
        // the guarantee that frames do not pile up in RAM. Now inside the
        // connected path, where it actually has something to do.
        client.flush();
    }

    // Running tally, so a bus that is degrading slowly is visible without
    // having to stare at the trace.
    if (badFrames > 0) {
        Serial.print(" (dropped ");
        Serial.print(badFrames);
        Serial.print(")");
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
