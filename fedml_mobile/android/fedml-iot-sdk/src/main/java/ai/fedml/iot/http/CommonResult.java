package ai.fedml.iot.http;

/**
 * Created by hechaoyang on 1/20/18.
 */

public class CommonResult {
    public static final int ERROR_CODE_SUCCESS = 0;//成功
    public static final int ERROR_CODE_SERVERERR = 1;//服务器本身错误
    public static final int ERROR_CODE_NOTAUTHEDERR = 2;//参数错误
    public static final int ERROR_CODE_PARAMERR = 3;//参数错误
    public static final int ERROR_CODE_DECODEERR = 4;//解密失败
    public static final int ERROR_CODE_AUTHERR = 5;//认证时发生的错误
    public static final int ERROR_CODE_CHECKERR = 6;//数据不完整错误，比如crc错误
    public static final int ERROR_CODE_NETERR = 7;//网络错误
    public static final int ERROR_CODE_UPDATEERR = 8;//更新错误
    public static final int ERROR_CODE_DB_ERR = 9;//mongo错误
    public static final int ERROR_CODE_DATA_EXISTED = 10;//数据已经存在
    public static final int ERROR_CODE_LOGINERR = 11;//登录错误
    public static final int ERROR_CODE_NOTEXIST = 12;//数据不存在错误
    public static final int ERROR_CODE_OTHERERR = 100;//其他错误

    /**
     * ErrorCode : 10
     * ErrorMsg : Data already exists and cannot be recreated. This IOT_UUID has already been registered!
     */

    private int ErrorCode;
    private String ErrorMsg;

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

    @Override
    public String toString() {
        return "CommonResult{" +
                "ErrorCode=" + ErrorCode +
                ", ErrorMsg='" + ErrorMsg + '\'' +
                '}';
    }
}
