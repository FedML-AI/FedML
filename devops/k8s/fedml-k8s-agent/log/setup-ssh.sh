#!/bin/bash
set -e  # 遇到错误立即退出

echo "[$(date)] Starting SSH installation..."

# 安装SSH服务
apt-get update
apt-get install -y openssh-server net-tools

# 验证安装
if [ ! -f /usr/sbin/sshd ]; then
    echo "Error: sshd not installed properly"
    exit 1
fi

# 确保服务目录存在
mkdir -p /var/run/sshd

# 配置SSH
mkdir -p /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIITAoK1DEXdsyB2UNZvZ5gPc0EJpn2V+gFLQTj18HsOz fedml@tensoropera.ai" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
chmod 700 /root/.ssh

# 配置SSH服务器
cat > /etc/ssh/sshd_config <<EOF
Port 22
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding yes
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
LogLevel INFO
EOF

chmod 600 /etc/ssh/sshd_config
# service ssh start || /usr/sbin/sshd

echo "[$(date)] SSH installation completed" 