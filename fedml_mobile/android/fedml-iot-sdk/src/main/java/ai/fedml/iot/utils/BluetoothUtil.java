package ai.fedml.iot.utils;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;

import net.vidageek.mirror.dsl.Mirror;

import java.util.Iterator;
import java.util.Set;

public class BluetoothUtil {

    public static String getBtMacAddress() {
        try{
            BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
            Object bluetoothManagerService = new Mirror().on(bluetoothAdapter).get().field("mService");
            if (bluetoothManagerService == null) {
                LogUtils.d("can not get Bluetooth MAC address");
                return "";
            }

            Object address = new Mirror().on(bluetoothManagerService).invoke().method("getAddress").withoutArgs();
            if(address != null && address instanceof String){
                return (String)address;
            }else{
                return "";
            }
        } catch (Exception e){
            return "";
        }
    }

    public static String getSavedPhoneBluetoothMac(){
        //获取已经保存过的设备信息
        BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
        Set<BluetoothDevice> devices = bluetoothAdapter.getBondedDevices();
        if (devices.size()>0) {
            for(Iterator<BluetoothDevice> iterator = devices.iterator(); iterator.hasNext();){
                BluetoothDevice bluetoothDevice = iterator.next();
                LogUtils.d("设备："+bluetoothDevice.getName() + " " + bluetoothDevice.getAddress());
            }
        }
        return null;
    }
}
