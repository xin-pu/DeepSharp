using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class DQNTest : AbstractTest
    {
        public DQNTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestDQN()
        {
            var frozenLake = new Frozenlake();
            var dqn = new DQN(frozenLake);
            var act = dqn.GetPolicyAct(frozenLake.Observation!.Value!);
            Print(act);
        }

        [Fact]
        public void KArmedBanditMain()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75}) {Gamma = 0.95f};
            var agent = new DQN(kArmedBandit, 100, 1000, batchSize: 16);
            Print(kArmedBandit);

            var i = 0;
            float reward;
            const int testEpisode = 20;
            const float predReward = 17f;
            do
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();

                reward = agent.TestEpisodes(testEpisode);
                Print($"{i:D5}:\t{reward}");
            } while (reward <= predReward);

            var episode = agent.RunEpisode();
            Print(episode);
        }


        [Fact]
        public void FrozenlakeMain()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
            var agent = new DQN(frozenlake, 100, 1000, 0.9f, batchSize: 16);
            Print(frozenlake);


            var i = 0;
            float reward;
            const int testEpisode = 20;
            const float predReward = 0.8f;
            do
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                reward = agent.TestEpisodes(testEpisode);
                Print($"{i:D5}:\t{reward}");
            } while (reward < predReward);

            Print($"Stop after Learn {i}");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}