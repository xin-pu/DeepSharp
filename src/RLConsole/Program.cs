// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using RLConsole;
using TorchSharp;

var testEpisode = 20;

/// Step 1 Create a 4-Armed Bandit
var frozenLake = new Frozenlake(deviceType: DeviceType.CPU) {Gamma = 0.9f};
Utility.Print(frozenLake);

/// Step 2 Create AgentQLearning
var agent = new QLearning(frozenLake, 1f);

/// Step 3 Learn and Optimize
var i = 0;
var total = 10000;
var bestReward = 0f;
while (i < total)
{
    i++;
    frozenLake.Reset();
    var epoch = 0;
    while (!frozenLake.IsComplete(epoch))
    {
        var action = agent.GetEpsilonAct(frozenLake.Observation!.Value!);
        var step = frozenLake.Step(action, epoch++);
        agent.Update(step);
    }

    agent.Gamma = (float) (1f / 2 * (1 + Math.Cos(i * Math.PI / total)));

    if (i % 50 == 0)
    {
        var episode = agent.RunEpisode(testEpisode);
        var reward = episode.Average(a => a.SumReward.Value);

        bestReward = new[] {bestReward, reward}.Max();
        Utility.Print($"{agent} Play:{i:D3}\t {reward}");
        if (bestReward > 0.81)
            break;
    }
}

frozenLake.ChangeToRough();
frozenLake.CallBack = s => { Utility.Print(frozenLake); };
var e = agent.RunEpisode();
var act = e.Steps.Select(a => a.Action);
Utility.Print(string.Join("\r\n", act));



