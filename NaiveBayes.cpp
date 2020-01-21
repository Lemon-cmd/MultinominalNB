#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <map>
#include <algorithm>

using namespace std;

class NaiveB
{
    private:
        //metadata for each state X/Y
        struct metadata
        {
            int column = NULL;
            int overall_freq = 0;
            double overall_prob = 0;
            double stateP = NULL;
        };   

        //likelihood of the edge X(i) -> Y(i)
        struct likelihood
        {
            int y_freq = 0;
            double y_prob = 0;
        };

        vector <vector<string> > train_data; // train data set
        vector <vector<string> > test_data;  // test data set
        vector <int> ignoredXs;             // vector that holds target column to be removed

        map <string, metadata*> Features; // X
        map <string, metadata*> Y;  // Y or outcomes

        map <string, map<string, likelihood*> > connections;    //use to store X(i) -> Y(i) connections
      
        int train_size;     //size of train set
        int test_size;      //size of test set

        void updateXY()
        {
            /* Method for Putting in features and outcomes into Features and Y Maps */

            for (auto &row : train_data)
            {
                int entry_size = row.size();        // current entry size; save performance on calling the size() method
                for (int col = 0; col < entry_size - 1; col ++)
                {
                    // if the feature is not in the map
                    if (Features.find(row[col]) == Features.end())
                    {
                        //set metadata and update overall freq of the feature
                        metadata* new_entryX = new metadata;
                        //set key (X) to connections map and also create its edge
                        map<string, likelihood*> edge;

                        new_entryX->overall_freq += 1;
                        new_entryX->column = col;           //save the column value of the feature
 
                        Features[row[col]] = new_entryX;    //set metadata
                        connections[row[col]] = edge;       //set edge to the current X

                        cout << "New X: " << row[col] << endl;
                    }
                    // if the feature is already in the map, update its count
                    else
                    {
                        Features[row[col]]->overall_freq += 1;
                    }
                }
                // if the outcome is not in the map
                if (Y.find(row[entry_size - 1]) == Y.end())
                {

                    // set Y metadata and update its freq
                    metadata* new_entryY = new metadata;
                    new_entryY->overall_freq += 1;
                    Y[row[entry_size - 1]] = new_entryY;

                    cout << "New Y: " << row[entry_size - 1] << endl;
                }
                // if outcome exists, update its count
                else
                {
                    Y[row[entry_size - 1]]->overall_freq += 1;
                }
            }
        }

        void clean(vector <vector<string> > &target)
        {
            //Method for cleaning unusuable features
            vector <vector <string> > new_data;
            
            //loop through the target vector
            for (auto &row : target)
            {
                vector <string> entry;          //vector for new row
                bool unusuable = false;         //bool value for checking row with no data

                for (int col = 0; col < row.size() - 1; col ++)
                {
                    //if there are targeted columns to be removed
                    if (ignoredXs.size() != 0)
                    {
                        if (find(ignoredXs.begin(), ignoredXs.end(), col) == ignoredXs.end())
                        {
                            entry.push_back(row[col]);
                        }   
                    }
                    //simply push all values if there are no targeted columns
                    else 
                    {
                        entry.push_back(row[col]);
                    }
                    
                    //a data in a column is blank 
                    if (row[col] == "?" || row[col] == " ")
                    {
                        unusuable = true;
                    }
                }
                //only push if entry contains no blank data
                if (unusuable == false)
                    new_data.push_back(entry);
            }

            //set y 
            for (int row = 0; row < target.size(); row ++)
            {
                new_data[row].push_back(target[row][target[row].size()-1]);
            }

            // set the old data vector to new data vector
            target = new_data;
        }

        void updateGlobalP()
        {
            /* A method for updating X and Y global probability; Target(i)'s freq / data size */
            //updating individual X global Probability
            for (auto &feature : Features)
            {
                feature.second->overall_prob = double(feature.second->overall_freq) / double(train_size);
            }

            //update individual Y global Probability
            for (auto &y : Y)
            {
                y.second->overall_prob = double(y.second->overall_freq) / double(train_size);
            }

        }

        void updateConnection()
        {
            //update connection between Feature and Y
            for (auto &con : connections)
            {
                for (auto &y : Y )
                {
                    for (auto &row : train_data)
                    {
                        int entry_size = row.size();

                        // y exists in the current row of data and the feature exists in that row
                        if (y.first == row[entry_size - 1] && row[Features[con.first]->column] == con.first)
                        {
                            // add edge to the connections map and update its freq
                            if (con.second.find(y.first) == con.second.end())
                            {
                                likelihood* item = new likelihood();
                                item->y_freq += 1;
                                con.second[y.first] = item;
                            }
                            // if already exists, simply update its freq
                            else
                            {
                                con.second[y.first]->y_freq += 1;
                            }
                        }
                        // if feature does not match with the y value in the row
                        else if (y.first != row[entry_size - 1] && row[Features[con.first]->column] == con.first)
                        {
                            //simply add the edge to the connection, Map(X(current))[y(current)] = item
                            if (con.second.find(y.first) == con.second.end())
                            {
                                likelihood* item = new likelihood();
                                con.second[y.first] = item;
                            }
                        }
                    }

                    // update edge probability
                    if (con.second[y.first]->y_freq != 0 && Features[con.first]->overall_freq != 0)
                    {
                        con.second[y.first]->y_prob = double(con.second[y.first]->y_freq) / double(Features[con.first]->overall_freq);
                    }
                    else 
                    {
                        con.second[y.first]->y_prob = 0.0;
                    }
                }
            } 
        }

        const string predictY(vector<string> &inputs)
        {
            // Prediction Method
            for (auto &y : Y)
            {   
                double equation = 0.0;
                //equation P(X(i) | Y) * P(Y)
                for (auto &inp : inputs)
                {
                    if (connections[inp].find(y.first) != connections[inp].end())
                    {
                        if (equation ==  0.0)
                        {
                            equation = connections[inp][y.first]->y_prob;
                        }
                        else
                        {
                            equation *= connections[inp][y.first]->y_prob; 
                        }
                    }
                }
                //set  y state probability
                y.second->stateP = double(equation); 
            }
            
            //get the best Y value
            string bestY;
            double bestYP = 0.0;

            //loop through Y map and basically find the maximum probability
            for (auto &y : Y)
            {
                if (bestYP < y.second->stateP)
                {
                    bestY = y.first; bestYP = y.second->stateP;
                }
            }

            //return the best Y with the best probability
            return bestY;    
        }

        void load(const string filename, vector <vector <string> > &target)
        {
      
            // load file onto the target vector
            ifstream file(filename);    // grab file
            string line;                // a string for holding  current line

            // loop through the file
            while (getline(file, line))
            {
                //if the current line is not empty
                if (!line.empty())
                {
                    //split by comma, or period
                    for (int c = 0; c < line.size(); c ++)
                    {
                        if (line[c] == ',' || line[c] == '.') line[c] = ' ';
                    }
                    
                    vector<string> row;    
                    istringstream input(line);
                    
                    //split by space
                    for (string s; input >> s;)
                    {
                        row.push_back(s);  //append individual string onto the vector
                    }
                    //insert the row onto the data
                    target.push_back(row);
                }
            }
        }   

    public:     
        NaiveB(vector <int> &removeXs) 
        {
            ignoredXs = removeXs;            // set ignoredXs vector
            
        }

        void loadTrainD(string traindata)
        {
            load(traindata, train_data);    // load train file
            clean(train_data);              // clean train data
            train_size = train_data.size();
        }

        void loadTestD(string testdata)
        {
            load(testdata, test_data);      // load test file
            clean(test_data);               // clean test data
            test_size = test_data.size();
        }

        void displayTrain()
        {
            //display the data vector
            for (auto &entry: train_data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }
         void displayTest()
        {
            //display the data vector
            for (auto &entry: test_data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }
        
        void predict()
        {
            // call update methods
            updateXY(); updateGlobalP(); updateConnection(); 
            int trues = 0;  //count for match Ys
            
            for (auto &row : test_data)
            {
                string current = predictY(row);         // get the best Y

                if (current == row[row.size()-1])       // check if the best Y matches with the test Y
                {
                    // increment if matches
                    trues += 1;
                }
            }
            // positives / total test size
            double accuracy = double(trues) / double(test_size);
            cout << "Accuracy of Model: " << accuracy * 100 << "%" << endl;
        }
};

int main()
{
    //push into the ignores vector for column/Feature that you want to eliminate
    //this is merely for data cleaning/cleansing
    vector <int> removeXs;
    removeXs.push_back(2);
    removeXs.push_back(4);
    removeXs.push_back(10);
    removeXs.push_back(11);
    removeXs.push_back(12);
    
    NaiveB classifier = NaiveB(removeXs);

    classifier.loadTrainD("adult.data");
    classifier.loadTestD("adult.test");

    classifier.predict();
   
    
}