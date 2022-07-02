package ai.fedml.widget;

import android.content.Context;
import android.graphics.drawable.ColorDrawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.PopupWindow;
import android.widget.TextView;

import ai.fedml.R;

/**
 * @创建者 xkai
 * @创建时间 2022/2/15 15:17
 * @描述
 */
public class PopupwindNormal extends PopupWindow {

    private View mView;
    private TextView tv_title;
    private TextView tv_content;

    private Button btn_cancel;
    private Button btn_ok;


    /**
     *
     * @param context 上下文
     * @param onClickListener 点击事件
     * @param title 标题
     * @param content 内容
     */
    public PopupwindNormal(Context context, View.OnClickListener onClickListener, String title, String content) {
        super(context);

        LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        mView = inflater.inflate(R.layout.popup_normal, null);

        tv_title = mView.findViewById(R.id.tv_title);
        tv_content = mView.findViewById(R.id.tv_content);
        btn_cancel = mView.findViewById(R.id.btn_cancel);
        btn_ok = mView.findViewById(R.id.btn_ok);

        btn_ok.setOnClickListener(onClickListener);
        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dismiss();
            }
        });

        tv_title.setText(title);
        tv_content.setText(content);




        // 设置PopupWindow的View
        this.setContentView(mView);
        // 设置PopupWindow弹出窗体的宽
        this.setWidth(ViewGroup.LayoutParams.MATCH_PARENT);
        // 设置PopupWindow弹出窗体的高
        this.setHeight(ViewGroup.LayoutParams.MATCH_PARENT);
        // 设置PopupWindow弹出窗体可点击
        this.setFocusable(true);
        // 设置PopupWindow弹出窗体动画效果
        this.setAnimationStyle(R.style.AnimFadePopup);
        // 实例化一个ColorDrawable颜色为黑色25%的透明度
        ColorDrawable dw = new ColorDrawable(0x40000000);
        // 设置PopupWindow弹出窗体的背景
        this.setBackgroundDrawable(dw);

    }

}
