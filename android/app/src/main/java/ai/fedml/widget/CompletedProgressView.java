package ai.fedml.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import ai.fedml.R;

/**
 * Custom circular progress bar
 */
public class CompletedProgressView extends View {

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
    // The x-coordinate of the center of the circle
    private int mXCenter;
    // The y coordinate of the center of the circle
    private int mYCenter;
    // word length
    private float mTxtWidth;
    // word height
    private float mTxtHeight;
    // total progress
    private int mTotalProgress = 100;
    // current progress
    private int mProgress;

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

    //draw
    @Override
    protected void onDraw(Canvas canvas) {
        mXCenter = getWidth() / 2;
        mYCenter = getHeight() / 2;

        //inner circle
        canvas.drawCircle(mXCenter, mYCenter, mRadius, mCirclePaint);

        //Outer arc background
        RectF oval1 = new RectF();
        oval1.left = (mXCenter - mRingRadius);
        oval1.top = (mYCenter - mRingRadius);
        oval1.right = mRingRadius * 2 + (mXCenter - mRingRadius);
        oval1.bottom = mRingRadius * 2 + (mYCenter - mRingRadius);
        canvas.drawArc(oval1, 0, 360, false, mRingPaintBg);

        //The ellipse object where the arc is located, the starting angle of the arc, the angle of the arc, whether to display the radius connection

        //Outer arc
        if (mProgress > 0 ) {
            RectF oval = new RectF();
            oval.left = (mXCenter - mRingRadius);
            oval.top = (mYCenter - mRingRadius);
            oval.right = mRingRadius * 2 + (mXCenter - mRingRadius);
            oval.bottom = mRingRadius * 2 + (mYCenter - mRingRadius);
            canvas.drawArc(oval, -90, ((float)mProgress / mTotalProgress) * 360, false, mRingPaint); //

            //fonts
            String txt = mProgress + "%";
            mTxtWidth = mTextPaint.measureText(txt, 0, txt.length());
            canvas.drawText(txt, mXCenter - mTxtWidth / 2, mYCenter + mTxtHeight / 4, mTextPaint);
        }
    }

    //set the progress
    public void setProgress(int progress) {
        mProgress = progress;
        postInvalidate();//redraw
    }


}
