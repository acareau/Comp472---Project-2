// Test driver for AND gate neural node
public class AndGateDriver {

    public static void main(String[] args){
        int x1, x2;
        // Create neural node object from AND gate class
        AndGate testGate = new AndGate();
        // Generate 10 sets of inputs, call object function to generate output and print results
        for (int i = 0; i < 10; i++) {
            x1 = (int)Math.round(Math.random());
            x2 = (int)Math.round(Math.random());
            System.out.println("The output of an AND gate with inputs A: " + x1 + " and B: " + x2 + " is ");
            System.out.println(testGate.generateOutput(x1, x2) + ".\n");
        }
    }
}
