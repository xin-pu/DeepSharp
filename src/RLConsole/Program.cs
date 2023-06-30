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
var frozenLake = new KArmedBandit(new[] {0.5d, 0.80d, 0.9d, 0.95d}) {Gamma = 0.9f};
Utility.Print(frozenLake);

/// Step 2 Create AgentQLearning
var agent = new DQN(frozenLake, 1, 500, gamma: 0.95f);

/// Step 3 Learn and Optimize
var i = 0;
var total = 1000;
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
        var reward = 1f * episode.Average(a => a.SumReward.Value);

        bestReward = new[] {bestReward, reward}.Max();
        Print($"{agent} Play:{i:D5}\t {reward}");
        //if (bestReward > 18)
        //    break;
    }
}

//frozenLake.ChangeToRough();
//frozenLake.CallBack = s => { Utility.Print(frozenLake); };
var e = agent.RunEpisode();
var act = e.Steps.Select(a => a.Action);
Print(string.Join("\r\n", act));