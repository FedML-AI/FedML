package ai.fedml.ui.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.List;

import ai.fedml.R;
import ai.fedml.utils.AppUtils;
import ai.fedml.utils.FormatUtils;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

/**
 * file list adapter
 */
public class RvFilePathAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    //Type, use this to determine which layout the recyclerview should use to display
    public final int TYPE_EMPTY = 0;
    public final int TYPE_NORMAL = 1;

    private final Context mContext;
    public List<FileItem> mSetData;


    public RvFilePathAdapter(Context mContext, List<FileItem> mSetData) {
        this.mContext = mContext;
        this.mSetData = mSetData;
    }

    @Override
    public int getItemViewType(int position) {
        if (mSetData == null || mSetData.size() <= 0) {
            return TYPE_EMPTY;
        }
        return TYPE_NORMAL;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view;
        //If it is an empty layout type, it will directly return null
        if (viewType == TYPE_EMPTY) {
            view = LayoutInflater.from(mContext).inflate(R.layout.rv_item_empty, parent, false);
            return new EmptyViewHolder(view);
        } else {
            view = LayoutInflater.from(mContext).inflate(R.layout.rv_item_file, parent, false);
            return new BodyViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        //First determine whether the holder is a custom holder
        if (holder instanceof BodyViewHolder) {
            holder.itemView.setOnClickListener((View v) -> {
                int pos = holder.getLayoutPosition();
                onItemClickListener.onItemClick(holder.itemView, pos);
            });
            if (mSetData == null) {
                return;
            }
            FileItem item = mSetData.get(position);
            if (item == null) {
                return;
            }
            ((BodyViewHolder) holder).img_icon.setImageResource(item.getFileIcon());
            ((BodyViewHolder) holder).tv_name.setText(item.getFileName());
            ((BodyViewHolder) holder).tv_time.setText(item.getFileLastModifiedTime());
            if (item.getFileIcon() == R.mipmap.ic_dir) {
                ((BodyViewHolder) holder).tv_size.setVisibility(View.GONE);
            } else {
                ((BodyViewHolder) holder).tv_size.setVisibility(View.VISIBLE);
                ((BodyViewHolder) holder).tv_size.setText(item.getFileSize());
            }
        }
    }

    @Override
    public int getItemCount() {
        if (mSetData == null || mSetData.size() <= 0) {
            return 1;
        }
        return mSetData.size();
    }

    public static class BodyViewHolder extends RecyclerView.ViewHolder {

        ImageView img_icon;
        TextView tv_name;
        TextView tv_size;
        TextView tv_time;

        public BodyViewHolder(@NonNull View itemView) {
            super(itemView);
            img_icon = itemView.findViewById(R.id.img_icon);
            tv_name = itemView.findViewById(R.id.tv_name);
            tv_size = itemView.findViewById(R.id.tv_size);
            tv_time = itemView.findViewById(R.id.tv_time);

        }
    }

    /**
     * empty layout
     */
    public static class EmptyViewHolder extends RecyclerView.ViewHolder {
        public EmptyViewHolder(@NonNull View itemView) {
            super(itemView);
        }
    }

    public interface OnItemClickListener {
        void onItemClick(View view, int position);
    }

    private OnItemClickListener onItemClickListener;

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
