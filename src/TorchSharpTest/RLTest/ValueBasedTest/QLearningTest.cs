using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class QLearningTest : AbstractTest
    {
        public QLearningTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void KArmedBanditMain()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75}) {Gamma = 0.95f};
            var agent = new QLearning(kArmedBandit);
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
            } while (reward < predReward);

            Print($"Stop after Learn {i}");

            var episode = agent.RunEpisode();
            Print(episode);
        }


        [Fact]
        public void FrozenlakeMain()
        {
            var frozenlake = new Frozenlake(new[] {0.6f, 0.2f, 0.2f}) {Gamma = 0.95f};
            var agent = new QLearning(frozenlake);
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

                reward = agent.TestEpisodes(testEpisode);
            } while (reward < predReward);

            Print($"Stop after Learn {i}");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
            ;
        }
    }
}