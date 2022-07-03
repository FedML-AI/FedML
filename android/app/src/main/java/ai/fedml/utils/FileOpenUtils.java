package ai.fedml.utils;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.StrictMode;

import java.io.File;

public class FileOpenUtils {
    private static final String[][] MIME_MapTable={
            //{suffix name, MIME type}
            {".3gp",    "video/3gpp"},
            {".apk",    "application/vnd.android.package-archive"},
            {".asf",    "video/x-ms-asf"},
            {".avi",    "video/x-msvideo"},
            {".bin",    "application/octet-stream"},
            {".bmp",    "image/bmp"},
            {".c",  "text/plain"},
            {".class",  "application/octet-stream"},
            {".conf",   "text/plain"},
            {".cpp",    "text/plain"},
            {".doc",    "application/msword"},
            {".docx",   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
            {".xls",    "application/vnd.ms-excel"},
            {".xlsx",   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
            {".exe",    "application/octet-stream"},
            {".gif",    "image/gif"},
            {".gtar",   "application/x-gtar"},
            {".gz", "application/x-gzip"},
            {".h",  "text/plain"},
            {".htm",    "text/html"},
            {".html",   "text/html"},
            {".jar",    "application/java-archive"},
            {".java",   "text/plain"},
            {".jpeg",   "image/jpeg"},
            {".jpg",    "image/jpeg"},
            {".js", "application/x-javascript"},
            {".log",    "text/plain"},
            {".m3u",    "audio/x-mpegurl"},
            {".m4a",    "audio/mp4a-latm"},
            {".m4b",    "audio/mp4a-latm"},
            {".m4p",    "audio/mp4a-latm"},
            {".m4u",    "video/vnd.mpegurl"},
            {".m4v",    "video/x-m4v"},
            {".mov",    "video/quicktime"},
            {".mp2",    "audio/x-mpeg"},
            {".mp3",    "audio/x-mpeg"},
            {".mp4",    "video/mp4"},
            {".mpc",    "application/vnd.mpohun.certificate"},
            {".mpe",    "video/mpeg"},
            {".mpeg",   "video/mpeg"},
            {".mpg",    "video/mpeg"},
            {".mpg4",   "video/mp4"},
            {".mpga",   "audio/mpeg"},
            {".msg",    "application/vnd.ms-outlook"},
            {".ogg",    "audio/ogg"},
            {".pdf",    "application/pdf"},
            {".png",    "image/png"},
            {".pps",    "application/vnd.ms-powerpoint"},
            {".ppt",    "application/vnd.ms-powerpoint"},
            {".pptx",   "application/vnd.openxmlformats-officedocument.presentationml.presentation"},
            {".prop",   "text/plain"},
            {".rc", "text/plain"},
            {".rmvb",   "audio/x-pn-realaudio"},
            {".rtf",    "application/rtf"},
            {".sh", "text/plain"},
            {".tar",    "application/x-tar"},
            {".tgz",    "application/x-compressed"},
            {".txt",    "text/plain"},
            {".wav",    "audio/x-wav"},
            {".wma",    "audio/x-ms-wma"},
            {".wmv",    "audio/x-ms-wmv"},
            {".wps",    "application/vnd.ms-works"},
            {".xml",    "text/plain"},
            {".z",  "application/x-compress"},
            {".zip",    "application/x-zip-compressed"},
            {"",        "*/*"}
    };

    /**
     * Obtain the corresponding MIME type according to the file suffix.
     * @param file
     */
    public static String getMIMEType(File file) {

        String type="*/*";
        String fName = file.getName();
        //Get the position of the separator "." before the suffix name in fName.
        int dotIndex = fName.lastIndexOf(".");
        if(dotIndex < 0){
            return type;
        }
        /* Get file extension */
        String end=fName.substring(dotIndex,fName.length()).toLowerCase();
        if(end=="")return type;
        //Find the corresponding MIME type in the MIME and file type match table.
        for(int i=0;i<MIME_MapTable.length;i++){
            if(end.equals(MIME_MapTable[i][0]))
                type = MIME_MapTable[i][1];
        }
        return type;
    }


    public static String getType(File file) {

        String type="";
        String fName = file.getName();
        //Get the position of the separator "." before the suffix name in fName.
        int dotIndex = fName.lastIndexOf(".");
        if(dotIndex < 0){
            return type;
        }
        /* Get file extension */
        String end=fName.substring(dotIndex,fName.length()).toLowerCase();
//        if(end=="")return type;
//        //Find the corresponding MIME type in the MIME and file type match table.
//        for(int i=0;i<MIME_MapTable.length;i++){
//            if(end.equals(MIME_MapTable[i][0]))
//                type = MIME_MapTable[i][0];
//        }
        return end;
    }

    /**
     * open the file
     * @param file
     */
    public static void openFile(Context context ,File file) {

        Intent intent = new Intent();
        if (Build.VERSION.SDK_INT>=Build.VERSION_CODES.N)
        {
            StrictMode.VmPolicy.Builder builder =  new StrictMode.VmPolicy.Builder();
            StrictMode.setVmPolicy(builder.build());
        }

//        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        //Set the Action property of the intent
        intent.setAction(Intent.ACTION_VIEW);
        //Get the MIME type of the file file
        String type = getMIMEType(file);
        //Set the data and Type properties of the intent.
        intent.setDataAndType(/*uri*/Uri.fromFile(file), type);
        //Jump
        context.startActivity(intent);

    }


    public static boolean  isImage(File  file)
    {
        String  type = getType(file);
        if (".jpg".equals(type)||".png".equals(type))
        {
            return true;
        }
        return false;

    }

}
