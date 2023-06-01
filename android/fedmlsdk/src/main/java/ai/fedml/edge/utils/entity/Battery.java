package ai.fedml.edge.utils.entity;

public class Battery {

    private String status;

    private String power;

    private String percentage;

    private String health;

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getPower() {
        return power;
    }

    public void setPower(String power) {
        this.power = power;
    }

    public String getPercentage() {
        return percentage;
    }

    public void setPercentage(String percentage) {
        this.percentage = percentage;
    }

    public String getHealth() {
        return health;
    }

    public void setHealth(String health) {
        this.health = health;
    }

    @Override
    public String toString() {
        return "Battery{" +
                "status='" + status + '\'' +
                ", power='" + power + '\'' +
                ", percentage='" + percentage + '\'' +
                ", health='" + health + '\'' +
                '}';
    }
}
