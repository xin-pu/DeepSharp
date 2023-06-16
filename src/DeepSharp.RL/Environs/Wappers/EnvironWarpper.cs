namespace DeepSharp.RL.Environs.Wappers
{
    public abstract class EnvironWarpper
    {
        protected EnvironWarpper(Environ<Space, Space> environ)
        {
            Environ = environ;
        }

        public Environ<Space, Space> Environ { set; get; }

        public abstract Step Step(Act act, int epoch);

        public abstract Observation Reset();
    }
}