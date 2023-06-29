// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using RLConsole;

static void Print(object obj)
{
    Console.WriteLine(obj.ToString());
}

var testEpisode = 100;

/// Step 1 Create a 4-Armed Bandit
var frozenLake = new Frozenlake(new[] {1f, 1f, 1f}) {Gamma = 0.9f};
Utility.Print(frozenLake);

/// Step 2 Create AgentQLearning
var agent = new QLearning(frozenLake, 1f);

/// Step 3 Learn and Optimize
var i = 0;
var total = 20000;
var bestReward = 0f;
while (i < total)
{
    i++;
    agent.Epsilon = 1f;
    frozenLake.Reset();
    var epoch = 0;
    while (true)
    {
        var action = agent.GetEpsilonAct(frozenLake.Observation!.Value!);
        var step = frozenLake.Step(action, epoch++);
        if (frozenLake.IsComplete(epoch))
            break;
        agent.Update(step);
    }

    if (i % 100 == 0)
    {
        agent.Epsilon = 0;
        var episode = agent.RunEpisode(testEpisode);
        var reward = 1f * episode.Count(a => a.SumReward.Value > 0) / testEpisode;

        bestReward = new[] {bestReward, reward}.Max();
        Print($"{agent} Play:{i:D3}\t {reward:P2}");
        if (bestReward > 0.5)
            break;
    }
}

frozenLake.ChangeToRough();
frozenLake.CallBack = s => { Utility.Print(frozenLake); };
var e = agent.RunEpisode();
var act = e.Steps.Select(a => a.Action);
Print(string.Join("\r\n", act));