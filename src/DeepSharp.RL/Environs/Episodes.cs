namespace DeepSharp.RL.Environs
{
    public class EpisodesSet
    {
        public EpisodesSet(Episode[] episodes)
        {
            Episodes = episodes;
        }

        public Episode[] Episodes { set; get; }
    }
}