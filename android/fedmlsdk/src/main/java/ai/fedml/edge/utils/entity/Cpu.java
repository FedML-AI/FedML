package ai.fedml.edge.utils.entity;

public class Cpu {

    private int cpuCores;

    private String cpuAbi;

    public int getCpuCores() {
        return cpuCores;
    }

    public void setCpuCores(int cpuCores) {
        this.cpuCores = cpuCores;
    }

    public String getCpuAbi() {
        return cpuAbi;
    }

    public void setCpuAbi(String cpuAbi) {
        this.cpuAbi = cpuAbi;
    }

    @Override
    public String toString() {
        return "CpuBean{" +
                "cpuCores=" + cpuCores +
                ", cpuAbi='" + cpuAbi + '\'' +
                '}';
    }
}
