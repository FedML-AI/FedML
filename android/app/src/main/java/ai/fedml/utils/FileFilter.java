package ai.fedml.utils;

import java.io.File;

/**
 * FileFilter
 *
 * @author xkai
 * @date 2022/1/5 11:02
 */
public class FileFilter implements java.io.FileFilter {

    @Override
    public boolean accept(File pathname) {
        // 过滤以“.”开头的文件和文件夹，获取文件名称，并且前缀是以“.”开头的，如果是返回：false，如果不是返回true
        if (pathname.getName().startsWith(".")) {
            return false;
        }
        return true;
    }
}
