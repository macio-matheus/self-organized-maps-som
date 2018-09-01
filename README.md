# Self organized maps SOM

The SOM network is a 2-layer neural network that accepts N-dimensional patterns as input and maps them to a set
    of output neurons, which represents the space of the data to be grouped. The output map, which is typically
    two-dimensional, represents the positions of the neurons in relation to their neighbors. The idea is that
    topologically close neurons respond in a similar way to similar inputs. For this, all input-layer neurons are all
    connected to the output neurons.

    SOM Training Algorithm Steps:
        1- Initialize the weights with random values.
        2- Display the input pattern for the network.
        3- Choose the output neuron with the highest activation state (Neurôno "winner").
        4- Update the weights of neighboring neurons to the winning neuron using a learning
        factor (Generally based on neighborhood radius and learning rate).
        5- Reduce the learning factor monotonically (linearly).
        6- Reduce the neighborhood radius monotonically (linearly).
        7- Repeat the steps from step 2 until the weight refreshes are very low.


### Build and run using Docker

```sh
cd self-organized-maps-som
docker-compose build
```

Run application 

```sh
cd self-organized-maps-som
docker-compose up
```