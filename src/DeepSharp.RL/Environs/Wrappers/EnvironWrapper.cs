namespace DeepSharp.RL.Environs.Wrappers
{
	public abstract class EnvironWrapper
	{
		protected EnvironWrapper(Environ<Space, Space> environ)
		{
			Environ = environ;
		}

		public Environ<Space, Space> Environ { get; set; }

		public abstract Step Step(Act act, int epoch);

		public abstract Observation Reset();
	}
}