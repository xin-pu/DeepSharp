// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using RLConsole;
using TorchSharp;

Console.WriteLine("Hello, World!");

var epoch = 10000;
var episodesEachBatch = 100;

/// Step 1 Create a 4-Armed Bandit
var forFrozenLake = new Frozenlake(DeviceType.CUDA)
{
    Gamma = 0.90f
};
Utility.Print(forFrozenLake);

/// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
var agent = new AgentCrossEntropyExt(forFrozenLake);

/// Step 3 Learn and Optimize
foreach (var i in Enumerable.Range(0, epoch))
{
    var batch = forFrozenLake.GetMultiEpisodes(agent, episodesEachBatch);
    var success = batch.Count(a => a.SumReward.Value > 0);

    var eliteOars = agent.GetElite(batch); /// Get eliteOars 

    /// Agent Learn by elite observation & action
    var loss = agent.Learn(eliteOars);
    var rewardMean = batch.Select(a => a.SumReward.Value).Sum();

    Utility.Print($"Epoch:{i:D4}\t:\t{success}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
}