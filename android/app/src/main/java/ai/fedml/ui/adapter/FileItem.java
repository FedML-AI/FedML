package ai.fedml.ui.adapter;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class FileItem {
    private int fileIcon;
    private String fileName;
    private String fileSize;
    private String fileLastModifiedTime;
}
