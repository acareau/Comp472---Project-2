// Neural node class definition for AND gate object
public class AndGate {
    // Two input variables
    int x1, x2;
    // Weights and bias preset for AND gate behaviour
    int w1 = 1, w2 = 1, b = -1;

    public int generateOutput(int x1, int x2) {
        // Correct for non-binary inputs
        if (x1 > 1) x1 = 1;
        if (x2 > 1) x2 = 1;
        // Output is 1 only if both inputs are 1
        return (x1*w1 + x2*w2 + b) > 0 ? 1 : 0;
    }
}
