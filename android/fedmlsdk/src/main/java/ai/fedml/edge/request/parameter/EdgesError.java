package ai.fedml.edge.request.parameter;

public class EdgesError {

    private String errMsg;

    private Integer errLine;

    public String getErrMsg() {
        return errMsg;
    }

    public void setErrMsg(String errMsg) {
        this.errMsg = errMsg;
    }

    public Integer getErrLine() {
        return errLine;
    }

    public void setErrLine(Integer errLine) {
        this.errLine = errLine;
    }
}
