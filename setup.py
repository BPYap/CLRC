import setuptools

setuptools.setup(
    name="clrc",
    version="0.1",
    packages=[
        "clrc",
        "clrc.data",
        "clrc.model", "clrc.model.encoder", "clrc.model.upstream", "clrc.model.downstream"
    ]
)