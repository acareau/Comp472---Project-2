public class AndGate {
    int x1, x2;
    int w1 = 1, w2 = 1, b = -1;

    public int generateOutput(int x1, int x2) {
        return (x1*w1 + x2*w2 + b) > 0 ? 1 : 0;
    }
}
