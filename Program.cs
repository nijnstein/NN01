using NN01.Tests;

do
{
    new LogicGate4Way().Run();
    Console.WriteLine("");
    Console.WriteLine("");
    
    new Pattern256x8().Run();

    Console.WriteLine("");
    Console.WriteLine("");
    
    Console.WriteLine("[SPACEBAR] to run tests again, any other to exit");
    Console.WriteLine("");
    Console.WriteLine("");
}
while (Console.ReadKey().Key == ConsoleKey.Spacebar);


