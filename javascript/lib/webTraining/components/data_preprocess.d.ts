export declare const IMAGE_W = 28;
export declare const IMAGE_H = 28;
export declare function fileToImg(file: File | Blob): Promise<{
    el: HTMLImageElement;
    src: string;
    name: string;
}>;
export declare function filesToImgs(files: File[] | Blob[]): Promise<{
    el: HTMLImageElement;
    src: string;
    name: string;
}[]>;
/**
 * @description generate uuid if the image need to be deleted
 */
export declare const guid: () => string;
