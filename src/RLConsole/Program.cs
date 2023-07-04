// See https://aka.ms/new-console-template for more information

using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

static void Print(object obj)
{
    Console.WriteLine(obj.ToString());
}

var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
var agent = new DQN(frozenlake, 100, 1000, 1f);
Print(frozenlake);


var i = 0;
float reward;
const int testEpisode = 20;
const float predReward = 0.7f;
do
{
    i++;

    frozenlake.Reset();
    agent.Learn();
    var e = 1f - (1 - 0.01f) / 1000 * i;
    agent.Epsilon = e < 0.01f ? 0.01f : e;
    reward = agent.TestEpisodes(testEpisode);
    Print($"{i}:\t{reward}");
} while (reward < predReward);

Print($"Stop after Learn {i}");
frozenlake.ChangeToRough();
var episode = agent.RunEpisode();
Print(episode);