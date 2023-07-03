using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using FluentAssertions;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class MonteCarloTest : AbstractTest
    {
        public MonteCarloTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void KABOnPolicyTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.80, 0.85, 0.80}) {Gamma = 0.95f};
            Print(kArmedBandit);

            var agent = new MonteCarloOnPolicy(kArmedBandit, 0.1f, 20);

            var i = 0;
            var testEpisode = 20;
            var bestReward = 0f;
            while (true)
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();


                var episode = agent.RunEpisodes(testEpisode);
                var reward = episode.Average(a => a.SumReward.Value);

                bestReward = new[] { bestReward, reward }.Max();
                Print($"{agent} Play:{i:D3}\t {reward}");
                if (bestReward > 17)
                    break;
            }

            var e = agent.RunEpisode();
            var act = e.Steps.Select(a => a.Action).ToList();
            Print(string.Join("\r\n", act));

            var bestResut = act.Select(a => a.Value!.ToInt32()).ToList();
            bestResut.All(a => a == 2).Should().BeTrue();
        }

        [Fact]
        public void KABOffPolicyTest()
        {
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.80, 0.85, 0.80}) {Gamma = 0.95f};

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new MonteCarloOffPolicy(kArmedBandit, 0.1f, 20);
            Print(kArmedBandit);

            var i = 0;
            var testEpisode = 20;
            var bestReward = 0f;
            while (true)
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();


                var episode = agent.RunEpisodes(testEpisode);
                var reward = episode.Average(a => a.SumReward.Value);

                bestReward = new[] {bestReward, reward}.Max();
                Print($"{agent} Play:{i:D3}\t {reward}");
                if (bestReward > 17)
                    break;
            }

            var e = agent.RunEpisode();
            var act = e.Steps.Select(a => a.Action);
            Print(string.Join("\r\n", act));
        }

        [Fact]
        public void FLOnPolicyTest()
        {
            var frozenlake = new Frozenlake(new[] {1f, 0f, 0f}) {Gamma = 0.9f};
            Print(frozenlake);

            var agent = new MonteCarloOnPolicy(frozenlake, 0.1f, 50);


            var i = 0;
            var testEpisode = 20;
            var bestReward = 0f;

            while (true)
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                if (i % 100 == 0)
                {
                    var episode = agent.RunEpisodes(testEpisode);
                    var reward = episode.Average(a => a.SumReward.Value);

                    bestReward = new[] {bestReward, reward}.Max();
                    Print($"{agent} Play:{i:D5}\t {reward}");
                    if (bestReward >= 0.5f)
                        break;
                }
            }

            frozenlake.ChangeToRough();
            frozenlake.CallBack = _ => { Print(frozenlake); };
            var e = agent.RunEpisode();
            var act = e.Steps.Select(a => a.Action).ToList();
            Print(string.Join("\r\n", act));

            var bestPath = act.Select(a => a.Value!.ToInt32()).ToList();

            bestPath.Zip(new[] {1, 1, 3, 1, 3, 3}, (a, b) => a == b)
                .All(a => a)
                .Should().BeTrue();
        }

        [Fact]
        public void FLOffPolictTest()
        {
            /// Step 1 Create a 4-Armed Bandit
            var frozenlake = new Frozenlake(new[] {0.6f, 0.2f, 0.2f}) {Gamma = 0.9f};

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new MonteCarloOffPolicy(frozenlake, 0.5f, 50);
            Print(frozenlake);

            var i = 0;
            var testEpisode = 20;
            var bestReward = 0f;

            while (true)
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                if (i % 100 == 0)
                {
                    var episode = agent.RunEpisodes(testEpisode);
                    var reward = episode.Average(a => a.SumReward.Value);

                    bestReward = new[] {bestReward, reward}.Max();
                    Print($"{agent} Play:{i:D3}\t {reward}");
                    if (bestReward > 0.5)
                        break;
                }
            }

            frozenlake.ChangeToRough();
            frozenlake.CallBack = _ => { Print(frozenlake); };
            var e = agent.RunEpisode();
            var act = e.Steps.Select(a => a.Action);
            Print(string.Join("\r\n", act));
        }
    }
}