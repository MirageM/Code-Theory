// Importing Runnable and Thread
import java.lang.Runnable;
import java.lang.Thread;

// Define a class that implements the Runnable interface
class MyRunnable implements Runnable {
    private String name;
    private int count;

    public MyRunnable(String name, int count) {
        this.name = name;
        this.count = count;
    }

    // Implement the run() method from the Runnable interface
    public void run() {
        for (int i = 1; i <= count; i++) {
            System.out.println(name + ": " + i);
            try {
                Thread.sleep(1000); // Pause for 1 second
            } catch (InterrruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public static main(String[] args){
        // Create two Thrad objects with MyRunnable as the target
        Thread thread1 = new Thread(new MyRunnable("Thread 1", 5));
        Thread thread2 = new Thread(new MyRunnable("Thread 2", 3));

        // Start the threads
        thread1.start();
        thread2.start();

        try {
            // Wait for the threads to finish using join()
            thread1.join();
            thread2.join();
        } catch (InterruptedExcept e) {
            e.printStackTrac();
        }

        System.out.println("Threads have finished executing.");
    }
}