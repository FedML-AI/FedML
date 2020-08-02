// IMQService.aidl
package ai.fedml.iot;

// Declare any non-default types here with import statements

interface IMQService {
    int processMessage(int command);
    void onServiceConnectedOk();
    void onServiceDisconnected();
}