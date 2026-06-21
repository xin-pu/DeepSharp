namespace TorchSharpTest.DemoTest
{
	/// <summary>
	///     This is demo net to guide how to create a new Module
	/// </summary>
	public sealed class DemoNet : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> layers;

		public DemoNet(int obsSize, int actionNum) : base("Net")
		{
			var modules = new List<(string, Module<Tensor, Tensor>)>
			{
				("line1", Linear(obsSize, 10)),
				("line2", Linear(10, actionNum))
			};
			layers = Sequential(modules);
			RegisterComponents();
		}


		public override Tensor forward(Tensor input)
		{
			return layers.forward(input);
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				layers.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}