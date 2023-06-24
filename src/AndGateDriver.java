public class AndGateDriver {

    public static void main(String[] args){
        int x1, x2;
        AndGate testGate = new AndGate();
        for (int i = 0; i < 10; i++) {
            x1 = (int)Math.round(Math.random());
            x2 = (int)Math.round(Math.random());
            System.out.println("The output of an AND gate with inputs A: " + x1 + " and B: " + x2 + " is ");
            System.out.println(testGate.generateOutput(x1, x2) + ".\n");
        }
    }
}
