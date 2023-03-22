package ai.fedml.edgedemo.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;

import ai.fedml.edgedemo.R;

/**
 * Custom circular progress bar
 */
public class CompletedProgressView extends View {

    // total progress
    private static final int TOTAL_PROGRESS = 100;
    // Paintbrush for drawing a filled circle
    private Paint mCirclePaint;
    // brush for drawing circles
    private Paint mRingPaint;
    // The background color of the brush for drawing the ring
    private Paint mRingPaintBg;
    // brush for drawing fonts
    private Paint mTextPaint;
    // circle color
    private int mCircleColor;
    // ring color
    private int mRingColor;
    // Ring background color
    private int mRingBgColor;
    // radius
    private float mRadius;
    // Ring radius
    private float mRingRadius;
    // Ring width
    private float mStrokeWidth;
    // word height
    private float mTxtHeight;
    // current progress
    private int mProgress;
    private RectF mOuterRect;
    private String mStatus;

    public CompletedProgressView(Context context, AttributeSet attrs) {
        super(context, attrs);
        // Get custom properties
        initAttrs(context, attrs);
        initVariable();
    }

    //properties
    private void initAttrs(Context context, AttributeSet attrs) {
        TypedArray typeArray = context.getTheme().obtainStyledAttributes(attrs,
                R.styleable.TasksCompletedView, 0, 0);
        mRadius = typeArray.getDimension(R.styleable.TasksCompletedView_radius, 80);
        mStrokeWidth = typeArray.getDimension(R.styleable.TasksCompletedView_strokeWidth, 10);
        mCircleColor = typeArray.getColor(R.styleable.TasksCompletedView_circleColor, 0xFFFFFFFF);
        mRingColor = typeArray.getColor(R.styleable.TasksCompletedView_ringColor, 0xFFFFFFFF);
        mRingBgColor = typeArray.getColor(R.styleable.TasksCompletedView_ringBgColor, 0xFFFFFFFF);
        mProgress = typeArray.getInteger(R.styleable.TasksCompletedView_progress, 0);

        mRingRadius = mRadius + mStrokeWidth / 2;
    }

    //Initialize brush
    private void initVariable() {
        //inner circle
        mCirclePaint = new Paint();
        mCirclePaint.setAntiAlias(true);
        mCirclePaint.setColor(mCircleColor);
        mCirclePaint.setStyle(Paint.Style.FILL);

        //Outer arc background
        mRingPaintBg = new Paint();
        mRingPaintBg.setAntiAlias(true);
        mRingPaintBg.setColor(mRingBgColor);
        mRingPaintBg.setStyle(Paint.Style.STROKE);
        mRingPaintBg.setStrokeWidth(mStrokeWidth);


        //Outer arc
        mRingPaint = new Paint();
        mRingPaint.setAntiAlias(true);
        mRingPaint.setColor(mRingColor);
        mRingPaint.setStyle(Paint.Style.STROKE);
        mRingPaint.setStrokeWidth(mStrokeWidth);
        //mRingPaint.setStrokeCap(Paint.Cap.ROUND);//Set the line style, there are circles and squares

        //middle word
        mTextPaint = new Paint();
        mTextPaint.setAntiAlias(true);
        mTextPaint.setStyle(Paint.Style.FILL);
        mTextPaint.setColor(getResources().getColor(R.color.color_3C4043));
        mTextPaint.setTextSize(mRadius / 2);

        Paint.FontMetrics fm = mTextPaint.getFontMetrics();
        mTxtHeight = (int) Math.ceil(fm.descent - fm.ascent);
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
    }

    //draw
    @Override
    protected void onDraw(Canvas canvas) {
        // The x-coordinate of the center of the circle
        int mXCenter = getWidth() / 2;
        // The y coordinate of the center of the circle
        int mYCenter = getHeight() / 2;

        //inner circle
        canvas.drawCircle(mXCenter, mYCenter, mRadius, mCirclePaint);

        //Outer arc background
        if (mOuterRect == null) {
            mOuterRect = new RectF();
            mOuterRect.left = (mXCenter - mRingRadius);
            mOuterRect.top = (mYCenter - mRingRadius);
            mOuterRect.right = mRingRadius * 2 + (mXCenter - mRingRadius);
            mOuterRect.bottom = mRingRadius * 2 + (mYCenter - mRingRadius);
        }
        canvas.drawArc(mOuterRect, 0, 360, false, mRingPaintBg);

        //The ellipse object where the arc is located, the starting angle of the arc, the angle of the arc, whether to display the radius connection

        //Outer arc
        if (mProgress > 0) {
            canvas.drawArc(mOuterRect, -90, ((float) mProgress / TOTAL_PROGRESS) * 360, false, mRingPaint); //
        }

        //fonts
        String txt = mStatus;
        if (TextUtils.isEmpty(mStatus) && mProgress > 0) {
            txt = mProgress + "%";
        }
        if (!TextUtils.isEmpty(txt)) {
            // word length
            float mTxtWidth = mTextPaint.measureText(txt, 0, txt.length());
            canvas.drawText(txt, mXCenter - mTxtWidth / 2, mYCenter + mTxtHeight / 4, mTextPaint);
        }
    }

    //set the progress
    public void setProgress(int progress) {
        mProgress = progress;
        mStatus = null;
        postInvalidate();
    }

    /**
     * Set text status
     *
     * @param status status
     */
    public void setStatus(String status) {
        mStatus = status;
        postInvalidate();
    }
}
