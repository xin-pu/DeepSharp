// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using RLConsole;
using TorchSharp;

var episodesEachBatch = 100;
var testEpisode = 20;

/// Step 1 Create a 4-Armed Bandit
var frozenLake = new Frozenlake(deviceType: DeviceType.CPU) {Gamma = 0.9f};
Utility.Print(frozenLake);

/// Step 2 Create AgentQLearning
var agent = new AgentQLearning(frozenLake);

/// Step 3 Learn and Optimize
var i = 0;

var bestReward = 0f;
while (true)
{
    agent.Learn(episodesEachBatch);

    var episode = agent.PlayEpisode(testEpisode, updateAgent: true);

    var reward = episode.Count(a => a.SumReward.Value > 0) * 1f / testEpisode;
    bestReward = new[] {bestReward, reward}.Max();
    Utility.Print($"{agent} Play:{++i:D3}\t {bestReward:P2}");
    if (bestReward >= 0.75)
        break;
}

frozenLake.ChangeToRough();
frozenLake.CallBack = s => { Utility.Print(frozenLake); };

var e = agent.PlayEpisode();

var act = e.Steps.Select(a => a.Action);
Utility.Print(string.Join("\r\n", act));



