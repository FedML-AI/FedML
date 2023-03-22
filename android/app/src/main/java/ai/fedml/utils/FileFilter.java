package ai.fedml.utils;

import java.io.File;

/**
 * FileFilter
 */
public class FileFilter implements java.io.FileFilter {

    @Override
    public boolean accept(File pathname) {
        // Filter files and folders starting with ".", get the file name, and the prefix is ".", if it is, return false, if not return true
        if (pathname.getName().startsWith(".")) {
            return false;
        }
        return true;
    }
}
