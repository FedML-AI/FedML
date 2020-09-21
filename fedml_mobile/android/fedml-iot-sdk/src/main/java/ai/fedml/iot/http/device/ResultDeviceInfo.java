package ai.fedml.iot.http.device;


public class ResultDeviceInfo<T> {
    /**
     * ErrorCode : 0
     * ErrorMsg : Successfully.
     * DeviceInfo : deviceinfo
     */
    private int ErrorCode;
    private String ErrorMsg;
    private T DeviceInfo;

    public int getErrorCode() {
        return ErrorCode;
    }

    public void setErrorCode(int ErrorCode) {
        this.ErrorCode = ErrorCode;
    }

    public String getErrorMsg() {
        return ErrorMsg;
    }

    public void setErrorMsg(String ErrorMsg) {
        this.ErrorMsg = ErrorMsg;
    }

    public T getDeviceInfo() {
        return DeviceInfo;
    }

    public void setDeviceInfo(T DeviceInfo) {
        this.DeviceInfo = DeviceInfo;
    }

    @Override
    public String toString() {
        return "ResultDeviceInfo{" +
                "ErrorCode=" + ErrorCode +
                ", ErrorMsg='" + ErrorMsg + '\'' +
                ", DeviceInfo=" + DeviceInfo +
                '}';
    }
}
