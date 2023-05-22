package ai.fedml.edge.utils.entity;

public class Memory {

    private String ramMemoryTotal;

    private String ramMemoryAvailable;

    /**
     * rom Available
     */
    private String romMemoryAvailable;

    /**
     * rom total
     */
    private String romMemoryTotal;

    public String getRamMemoryTotal() {
        return ramMemoryTotal;
    }

    public void setRamMemoryTotal(String ramMemoryTotal) {
        this.ramMemoryTotal = ramMemoryTotal;
    }

    public String getRamMemoryAvailable() {
        return ramMemoryAvailable;
    }

    public void setRamMemoryAvailable(String ramMemoryAvailable) {
        this.ramMemoryAvailable = ramMemoryAvailable;
    }

    public String getRomMemoryAvailable() {
        return romMemoryAvailable;
    }

    public void setRomMemoryAvailable(String romMemoryAvailable) {
        this.romMemoryAvailable = romMemoryAvailable;
    }

    public String getRomMemoryTotal() {
        return romMemoryTotal;
    }

    public void setRomMemoryTotal(String romMemoryTotal) {
        this.romMemoryTotal = romMemoryTotal;
    }

    @Override
    public String toString() {
        return "Memory{" +
                "ramMemoryTotal='" + ramMemoryTotal + '\'' +
                ", ramMemoryAvailable='" + ramMemoryAvailable + '\'' +
                ", romMemoryAvailable='" + romMemoryAvailable + '\'' +
                ", romMemoryTotal='" + romMemoryTotal + '\'' +
                '}';
    }
}
