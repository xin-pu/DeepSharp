// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using RLConsole;

static void Print(object obj)
{
    Console.WriteLine(obj.ToString());
}

var testEpisode = 20;

/// Step 1 Create a 4-Armed Bandit
var frozenLake = new Frozenlake(new[] {1f, 1f, 1f}) {Gamma = 0.9f};
Utility.Print(frozenLake);

/// Step 2 Create AgentQLearning
var agent = new SARSA(frozenLake, 1E-1f);

/// Step 3 Learn and Optimize
var i = 0;
var total = 5000;
var bestReward = 0f;
while (i < total)
{
    i++;
    agent.Epsilon = 1f;
    frozenLake.Reset();
    agent.Learn();

    if (i % 20 == 0)
    {
        var episode = agent.RunEpisodes(testEpisode);
        var reward = 1f * episode.Count(a => a.SumReward.Value > 0) / testEpisode;

        bestReward = new[] {bestReward, reward}.Max();
        Print($"{agent} Play:{i:D5}\t {reward:P2}");
        if (bestReward > 0.8)
            break;
    }
}

frozenLake.ChangeToRough();
frozenLake.CallBack = s => { Utility.Print(frozenLake); };
var e = agent.RunEpisode();
var act = e.Steps.Select(a => a.Action);
Print(string.Join("\r\n", act));