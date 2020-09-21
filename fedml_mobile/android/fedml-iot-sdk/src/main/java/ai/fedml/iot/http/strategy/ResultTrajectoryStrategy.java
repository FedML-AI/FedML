package ai.fedml.iot.http.strategy;

/**
 * Created by hechaoyang on 1/20/18.
 */

public class ResultTrajectoryStrategy<T> {

    /**
     * ErrorCode : 0
     * ErrorMsg : Successfully.
     * strategy : strategy JSON object
     */

    private int ErrorCode;
    private String ErrorMsg;
    private T strategy;

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

    public T getStrategy() {
        return strategy;
    }

    public void setStrategy(T strategy) {
        this.strategy = strategy;
    }

    @Override
    public String toString() {
        return "ResultTrajectoryStrategy{" +
                "ErrorCode=" + ErrorCode +
                ", ErrorMsg='" + ErrorMsg + '\'' +
                ", strategy=" + strategy +
                '}';
    }
}