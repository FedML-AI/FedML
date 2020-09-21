package ai.fedml.iot.http.strategy;

public class TrajectoryStrategy {

    /**
     * IOT_UUID : 60e1bef0f0676d73365b74c67ffe8b1d
     * AppServerDomain : http://111.230.226.28
     * DataServerDomain : tcp://mq.iotsupercloud.com:1883
     * DataServerAdmin : admin
     * DataServerPassword : admin
     * TrajectorySwitch : on
     * TrajectoryUpdateInterval : 0
     * TrajectoryUploadInterval : 0
     * Active : false
     * SpeechSwitch : off
     * PhoneBookSwitch : off
     */

    private String IOT_UUID;
    private String AppServerDomain;
    private String DataServerDomain;
    private String DataServerAdmin;
    private String DataServerPassword;
    private String TrajectorySwitch;
    private int TrajectoryUpdateInterval;
    private int TrajectoryUploadInterval;
    private String Active;
    private String SpeechSwitch;
    private String PhoneBookSwitch;

    public String getIOT_UUID() {
        return IOT_UUID;
    }

    public void setIOT_UUID(String IOT_UUID) {
        this.IOT_UUID = IOT_UUID;
    }

    public String getAppServerDomain() {
        return AppServerDomain;
    }

    public void setAppServerDomain(String AppServerDomain) {
        this.AppServerDomain = AppServerDomain;
    }

    public String getDataServerDomain() {
        return DataServerDomain;
    }

    public void setDataServerDomain(String DataServerDomain) {
        this.DataServerDomain = DataServerDomain;
    }

    public String getDataServerAdmin() {
        return DataServerAdmin;
    }

    public void setDataServerAdmin(String DataServerAdmin) {
        this.DataServerAdmin = DataServerAdmin;
    }

    public String getDataServerPassword() {
        return DataServerPassword;
    }

    public void setDataServerPassword(String DataServerPassword) {
        this.DataServerPassword = DataServerPassword;
    }

    public String getTrajectorySwitch() {
        return TrajectorySwitch;
    }

    public void setTrajectorySwitch(String TrajectorySwitch) {
        this.TrajectorySwitch = TrajectorySwitch;
    }

    public int getTrajectoryUpdateInterval() {
        return TrajectoryUpdateInterval;
    }

    public void setTrajectoryUpdateInterval(int TrajectoryUpdateInterval) {
        this.TrajectoryUpdateInterval = TrajectoryUpdateInterval;
    }

    public int getTrajectoryUploadInterval() {
        return TrajectoryUploadInterval;
    }

    public void setTrajectoryUploadInterval(int TrajectoryUploadInterval) {
        this.TrajectoryUploadInterval = TrajectoryUploadInterval;
    }

    public String getActive() {
        return Active;
    }

    public void setActive(String Active) {
        this.Active = Active;
    }

    public String getSpeechSwitch() {
        return SpeechSwitch;
    }

    public void setSpeechSwitch(String SpeechSwitch) {
        this.SpeechSwitch = SpeechSwitch;
    }

    public String getPhoneBookSwitch() {
        return PhoneBookSwitch;
    }

    public void setPhoneBookSwitch(String PhoneBookSwitch) {
        this.PhoneBookSwitch = PhoneBookSwitch;
    }

    @Override
    public String toString() {
        return "TrajectoryStrategy{" +
                "IOT_UUID='" + IOT_UUID + '\'' +
                ", AppServerDomain='" + AppServerDomain + '\'' +
                ", DataServerDomain='" + DataServerDomain + '\'' +
                ", DataServerAdmin='" + DataServerAdmin + '\'' +
                ", DataServerPassword='" + DataServerPassword + '\'' +
                ", TrajectorySwitch='" + TrajectorySwitch + '\'' +
                ", TrajectoryUpdateInterval=" + TrajectoryUpdateInterval +
                ", TrajectoryUploadInterval=" + TrajectoryUploadInterval +
                ", Active='" + Active + '\'' +
                ", SpeechSwitch='" + SpeechSwitch + '\'' +
                ", PhoneBookSwitch='" + PhoneBookSwitch + '\'' +
                '}';
    }
}