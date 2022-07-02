package ai.fedml.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;

import ai.fedml.R;
import androidx.core.content.ContextCompat;

/**
 * @创建者 xkai
 * @创建时间 2021/12/30 16:57
 * @描述 自定义的凹凸布局
 */
public class CouponTextView  extends androidx.appcompat.widget.AppCompatTextView {

    private Paint mPaint;
    private Context mContext;
    private int mColor;
    private int mHeight;
    private RectF mRectF;

    public CouponTextView(Context context) {
        this(context, null);

    }

    public CouponTextView(Context  context, AttributeSet attrs) {
        this(context, attrs, 0);

    }

    public CouponTextView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        TypedArray array = context.obtainStyledAttributes(attrs, R.styleable.CouponTextView);
        mColor = ContextCompat.getColor(context, R.color.white);
        mColor = array.getColor(R.styleable.CouponTextView_bg_color, mColor);
        mHeight = array.getDimensionPixelSize(R.styleable.CouponTextView_android_height, 30);
        mContext = context;
        initPaint();
        array.recycle();
    }

    private void initPaint() {
        mPaint =new Paint();
        mPaint.setColor(mColor);
        mPaint.setStrokeWidth(12f);
        mPaint.setAntiAlias(true);
    }

    @Override
    protected void onDraw(Canvas canvas) {

        super.onDraw(canvas);
        if(mRectF == null) {
            mRectF = new RectF(0, 0, getMeasuredWidth(), mHeight);
        }
        canvas.drawRect(mRectF, mPaint);
        mPaint.setColor(ContextCompat.getColor(mContext, R.color.color_F5F6FA));
        canvas.drawCircle(getMeasuredWidth()/2, 0,50, mPaint);
    }


}
