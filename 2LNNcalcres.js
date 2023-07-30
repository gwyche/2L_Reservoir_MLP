//CONTROL VARIABLES/////
//Input vector
var input = [.9,.7,.1,0,0,.1,0,.36];

var inputSize = 50;

//Vector the AI is training to match
var targetVector = [.5,0,0,(0-.5),0,0,.3,0];

//Hyperparameters
var learningFactor = .1;
var percentUpdated = .008;
var bias = .0001;
var reservoirPercentage = .1;
var reservoirGearRatio = 4;

//Loop iterations
var runs = 1000;
////////////////////////

/////////////////MAIN VARIABLES/////////////
var inputVector = [];

let errorVectorL1 = [];
let errorVectorL2 = [];

let errorGradientL2 = [];

let outputVector1 = [];
let outputVector2 = [];

let outputVectorA1 = [];
let outputVectorA2 = [];

let dOutput1 = [];

let transformedVector = [];
let transformedVectorTemp = [];
let gradientMatrix2 = createZMatrix(inputSize);
let gradientMatrix1 = createZMatrix(inputSize);
/////////////////END MAIN VARIABLES/////////////

inputVector = input;

////////PAD INPUT AND TARGET VECTORS////////
for(let i = input.length; i < inputSize; i++){
    inputVector[i] = 0;
}

for(let i = targetVector.length; i < inputSize; i++){
    targetVector[i] = 0;
}

///////END PADDING////////

////////////MATRIX GENERATION UTIL FUNCTIONS//////////////
//Matrix generator that randomizes all cells
function create0Matrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = Math.random() - .5;
        }
      }
    return matrix;
}

//Matrix generator that zeroes all cells
function createZMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = 0;
        }
      }
    return matrix;
}
////////////END MATRIX GENERATION UTIL FUNCTIONS//////////////

////////////MISC FUNCTIONS///////////////
function dActivation(x){
    return 4/(Math.pow((Math.exp((0-1)*x)+Math.exp(x)),2));
}
///////////END MISC FUNCTIONS///////////

function createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let j = 0; j < diameter; j++) {
            if(Math.random() < reservoirPercentage){
                matrix[i][j] = Math.random() - .5;
            }else{
                matrix[i][j] = 0;
            }
        }
      }
    return matrix;
}

///////////BEGIN FORWARD PASS////////////
//Matrix Multiplies Input Vector by Weight Matrix and Applies Standard Tanh Activation Function
function matrixMultiply1(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        outputVector1[i] = sum;
        outputVectorA1[i] = Math.tanh(sum + bias);
        outputVector[i] = Math.tanh(sum + bias);
    }
    return outputVector;
}

function matrixMultiply2(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        outputVector2[i] = sum;
        outputVectorA2[i] = Math.tanh(sum + bias);
        outputVector[i] = Math.tanh(sum + bias);
    }
    return outputVector;
}


///////////END FORWARD PASS///////////

////////////BACKWARD PASS UTIL FUNCTIONS//////////////


/////////////////RESERVOIR///////////////////<

//Reservoir computer preprocesses temporally correlated inputs
function updateReservoirProjection(inputV, weightM, diam, zeroIndex, initialInputProjection){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        if(i >= zeroIndex){
            inputV[i] = initialInputProjection[i] + inputV[i];
        }
    }
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let j = 0; j < diam; j++){
            sum = sum + (inputV[j] * weightM[i][j]);
        }
        outputVector[i] = Math.tanh(sum + bias);
    }
    initialInputProjection = outputVector;
    return outputVector;
}

function normalizeInputs(projectedInputs){
    var sum = 0;
    for(let i = 0; i < projectedInputs.length; i++){
        sum = sum + projectedInputs[i];
    }
    for(let i = 0; i < projectedInputs.length; i++){
        projectedInputs[i] = projectedInputs[i] / sum;
    }
    return projectedInputs;
}                                                                                                            


/////////////////RESERVOIR////////////////>

function backPropagateO(arg1, arg2, arg3, arg4){
    let dError = [];
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    bpWeights = arg1;
    dError = arg2;
    outV = arg3;
    dOutput = arg4;

    
    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            gradientMatrix2[i][j] = (learningFactor*outputVector1[j]*dOutput[i][j]*dError[i][j]);
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + gradientMatrix2[i][j];
        }
    }

    return bpWeights;
}

let dErrorVectr2 = createZMatrix(inputSize);
let dOutput2 = createZMatrix(inputSize);

function backPropagateH1(arg1, arg3, arg4, arg5){
    let dErrorMost = createZMatrix(inputSize);
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    let outputA = [];
    let E = 0;
    bpWeights = arg1;
    outV = arg3;
    dOutput = arg4;
    outputA = arg5;


    //Calculate dErrorMost//
    for(let i = 0; i < inputSize; i++){
        E = 0;
        for(let j = 0; j < inputSize; j++){
                E = E + weightMatrix2Transpose[i][j]*dErrorVectr2[i][j]*dOutput2[i][j];
            dErrorMost[i][j] = E;
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            gradientMatrix1[i][j] = (learningFactor*inputVector[i]*dOutput[j]*dErrorMost[i][j]);
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + gradientMatrix1[i][j];
        }
    }

    return bpWeights;
}

function createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            if(Math.random() < reservoirPercentage){
                matrix[i][k] = Math.random() - .5;
            }else{
                matrix[i][k] = 0;
            }
        }
      }
    return matrix;
}


////////////END BACKWARD PASS UTIL FUNCTIONS//////////////

//New randomized square weight matrix
let weightMatrix1 = create0Matrix(inputSize);
let weightMatrix2 = create0Matrix(inputSize);
let weightMatrix2Transpose = create0Matrix(inputSize);

let reservoirMatrix = createResMatrix(inputSize);
let projectedInputVector = [];
for(let i = 0; i < inputSize; i++){
    projectedInputVector[i] = 0;
}
let normalizedProjectedInput = [];


for(let r = 0; r < runs; r++){
    var errorInitial = 0;
    var tempError = 0;
    var finalError = 0;

    projectedInputVector = updateReservoirProjection(inputVector, reservoirMatrix, inputSize, input.length, projectedInputVector);
    normalizedProjectedInput = normalizeInputs(projectedInputVector);
    //console.log(normalizedProjectedInput);

    for(let a = 0; a < reservoirGearRatio; a++){

        transformedVector = matrixMultiply2(matrixMultiply1(inputVector, weightMatrix1, inputVector.length),weightMatrix2,inputVector.length);
        //console.log(transformedVector);

        for(let i = 0; i < inputSize; i++){
            dOutput1[i] = dActivation(outputVector1[i]);
        }

    
        for(let i = 0; i < inputSize; i++){
            for(let j = 0; j < inputSize; j++){
                dOutput2[i][j] = dActivation(outputVector2[i]);
            }
        }

    
        for(let k = 0; k < inputSize; k++){
            errorVectorL2[k] = ((Math.pow((targetVector[k] - transformedVector[k]),2))/2);
        }


        for(let i = 0; i < inputSize; i++){
            for(let j = 0; j < inputSize; j++){
                dErrorVectr2[i][j] = (targetVector[i] - transformedVector[i]);
            }
        }


        for(let k = 0; k < inputSize; k++){
            errorInitial = errorInitial + ((Math.pow((targetVector[k] - transformedVector[k]),2))/2);
        }

        //Backpropogate Weight Matrix 2
        weightMatrix2 = backPropagateO(weightMatrix2, dErrorVectr2, outputVectorA1, dOutput2);

        //Transpose Weight Matrix 2
        for(let s = 0; s < inputSize; s++){
            for(let t = 0; t < inputSize; t++){
                weightMatrix2Transpose[t][s] = weightMatrix2[s][t];
            }
        }

        //Backpropogate Weight Matrix 1
        weightMatrix1 = backPropagateH1(weightMatrix1, outputVector1, dOutput1, outputVectorA1);

        console.log("Error: "+errorInitial.toExponential());
    }
}
// console.log("");
// console.log("targ:"+targetVector);

