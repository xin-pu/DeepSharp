﻿using DeepSharp.RL.Agents;
using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.EnvironTest
{
    public class KArmedBanditTest : AbstractTest
    {
        public KArmedBanditTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void BanditCreateTest()
        {
            var count = 0;
            var bandit = new Bandit("A");
            Print(bandit);
            var range = 100;
            foreach (var _ in Enumerable.Range(0, range))
            {
                var res = bandit.Step();
                if (res > 0) count++;
            }

            Print(count * 1f / range);
        }

        [Fact]
        public void KArmedBanditCreateTest()
        {
            var kArmedBandit = new KArmedBandit(5);
            Print(kArmedBandit);
        }

        [Fact]
        public void RandomPickup()
        {
            var probs = new[] {0.5f, 0.5f};

            foreach (var _ in Enumerable.Repeat(0, 10))
            {
                var res = torch.multinomial(torch.from_array(probs), 1, true);
                var index = res.item<long>();
                Print(index);
            }
        }

        [Fact]
        public void AgentCrossEntropy()
        {
            var epoch = 100;
            var episodesEachBatch = 20;

            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(2)
            {
                [0] = {Prob = 0.4},
                [1] = {Prob = 0.75}
            };
            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new CrossEntropy(kArmedBandit, episodesEachBatch);

            /// Step 3 Learn and Optimize
            foreach (var i in Enumerable.Range(0, epoch))
            {
                var loss = agent.Learn();

                var test = agent.RunEpisodes(episodesEachBatch);

                var rewardMean = test.Select(a => a.SumReward.Value).Average();

                Print($"Epoch:{i:D4}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
            }
        }


        [Fact]
        public void QLearningRunRandom()
        {
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(2);
            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new ValueIteration(kArmedBandit, 20);
            agent.RunEpisodes(20, PlayMode.Sample);
            Print(kArmedBandit);
        }

        [Fact]
        public void QLearningMain()
        {
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(4)
            {
                [0] = {Prob = 0.5},
                [1] = {Prob = 0.2},
                [2] = {Prob = 0.4},
                [3] = {Prob = 0.8}
            };
            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new ValueIteration(kArmedBandit, 100);
            Print(kArmedBandit);

            var i = 0;
            var bestReward = 0f;
            while (i < 100)
            {
                agent.Learn();

                var episodes = agent.RunEpisodes(10);
                foreach (var episode in episodes)
                    agent.Update(episode);

                bestReward = new[] {bestReward, episodes.Average(a => a.SumReward.Value)}.Max();
                Print($"{agent} Play:{++i:D3}\t {bestReward}");
                if (bestReward > 18)
                    break;
            }
        }
    }
}