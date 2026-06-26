namespace RLSharp.Torch.Environs.Wrappers
{
	public abstract class EnvironWrapper
	{
		protected EnvironWrapper(EnvironmentBase<Space, Space> environment)
		{
			EnvironmentBase = environment;
		}

		public EnvironmentBase<Space, Space> EnvironmentBase { get; set; }

		public abstract Step Step(ActionValue action, int epoch);

		public abstract ObservationValue Reset();
	}
}