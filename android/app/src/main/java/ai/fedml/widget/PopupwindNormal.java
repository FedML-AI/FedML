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


public class PopupwindNormal extends PopupWindow {

    private View mView;
    private TextView tv_title;
    private TextView tv_content;

    private Button btn_cancel;
    private Button btn_ok;


    /**
     *
     * @param context
     * @param onClickListener
     * @param title
     * @param content
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




        // Set the View of PopupWindow
        this.setContentView(mView);
        // Set the width of the PopupWindow pop-up form
        this.setWidth(ViewGroup.LayoutParams.MATCH_PARENT);
        // Set the height of the PopupWindow pop-up form
        this.setHeight(ViewGroup.LayoutParams.MATCH_PARENT);
        // Set the PopupWindow pop-up form to be clickable
        this.setFocusable(true);
        // Set PopupWindow pop-up form animation effect
        this.setAnimationStyle(R.style.AnimFadePopup);
        // Instantiate a ColorDrawable with a color of black with 25% opacity
        ColorDrawable dw = new ColorDrawable(0x40000000);
        // Set the background of the PopupWindow popup form
        this.setBackgroundDrawable(dw);

    }

}
