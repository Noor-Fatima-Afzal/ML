public class regression{

    private double[] x;
    private double[] y;
    private double slope;
    private double intercept;

    // Constructor
    public regression(double[] x, double[] y){
        this.x = x;
        this.y = y;
        this.slope = 0;
        this.intercept = 0;
    }

    // Method to calculate the slope and intercept
    public void calculation(){
        int n = x.length;
        double sumX=0.0, sumY=0.0;

        // Calculating means of x and y 
               // first sum
        for (int i=0; i<n ;i++){
            sumX+=x[i];
            sumY+=y[i];
        }
               // now mean
        double xMean=sumX/n;
        double yMean=sumY/n;

        // now find numerator and denominator to find slope 
        double numerator = 0.0;
        double denominator = 0.0;
        for (int j=0; j<n ; j++){
            double xDiff = x[j]-xMean;
            double yDiff = y[j]-yMean;

            numerator += xDiff*yDiff;
            denominator += xDiff*xDiff;
        }

        slope = numerator / denominator;
        intercept = yMean - slope*xMean;
    }
    // method to predict y for given x
    public double predict(double xValue){
        return slope*xValue + intercept;
    }

    // Method to get slope 
    public double getSlope(){
        return slope;
    }
    // method to get intercept
    public double getIntercept(){
        return intercept;
    }

    public static void main(String[] args) {
        double[] x= {1,2,3,4,5};
        double[] y= {4,3,5,5,6,6};
        regression model1 = new regression(x, y);

        model1.calculation();

        System.out.println("slope is: "+ model1.getSlope());
        System.out.println("intercept is: "+ model1.getIntercept());

        double xValue=6;
        double predictedValue= model1.predict(xValue);
        System.out.println("Predicted y for x = " + xValue + ": " + predictedValue);
    }
}