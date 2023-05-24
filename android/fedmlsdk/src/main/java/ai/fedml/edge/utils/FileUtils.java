package ai.fedml.edge.utils;

import java.io.File;

public class FileUtils {
    public static boolean isEmptyDirectory(String path) {
        File directory = new File(path);
        if (directory.isDirectory()) {
            return directory.list().length==0;
        }
        else{
            return false;
        }
    }
}
