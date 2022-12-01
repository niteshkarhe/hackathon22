import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import {RestApiService} from '../../services/TransctionService';
import {Transaction} from '../../services/Transaction';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})

export class HomeComponent implements OnInit {
  loading = false;
  itemsList = [];
  showForm:boolean = true;
  showJSON:boolean = false;
  list_name;
  dataSource: FraudPrevention[] = [];
  transactionJSON;

  selectedType:string;
  amount:number;
  fromAccNo:string;
  destAccNo:string;
  oldbalanceOrg:number;


  transaction: Transaction;

  ITEMS = [{name: 1, value:'Single Transaction'}, {name: 2, value:'JSON'}];
  transTypes = ['CASH_IN', 'CASH_OUT' ,'DEBIT' ,'PAYMENT' ,'TRANSFER']

  obj = {'type': 'Type', 'amount':'Amount', 'fromAccNo':'Sender A/c No', 'destAccNo':'Receiver A/c No','oldbalanceOrg':'Sender Old Balance','newbalanceOrg':'Sender New Balance','prediction':'Prediction','decisionTreeAccuracy':'ML Accuracy','decisionTreePrediction':'ML Prediction','indication':'Indication'};
  displayedColumns = ['type', 'amount', 'fromAccNo', 'destAccNo','oldbalanceOrg','newbalanceOrg','prediction','decisionTreeAccuracy','decisionTreePrediction','indication'];
  columnsToDisplay: string[] = this.displayedColumns.slice();

  constructor(private http: HttpClient, private service : RestApiService) { }

  ngOnInit(): void {
    this.itemsList = this.ITEMS;
    this.showForm = true;
    this.showJSON = false;
    this.transactionJSON = '[{"type": "TRANSFER","amount": 18000.00,"fromAccNo": "C905679615","destAccNo": "C1023714065","oldbalanceOrg": 100000.00},{"type": "TRANSFER","amount": 8000.00,"fromAccNo": "C905679615","destAccNo": "CC572787442","oldbalanceOrg": 100000.00}]';
  }

  onSubmit(){

      this.transaction = {
        'type': this.selectedType,
        'amount': this.amount,
        'fromAccNo': this.fromAccNo,
        'destAccNo': this.destAccNo,
        'oldbalanceOrg': this.oldbalanceOrg
      };


        this.loading = true;

    this.http.post<FraudPrevention[]>("http://localhost:5000/fraud/predict", (this.showForm)?this.transaction:JSON.parse(this.transactionJSON)).subscribe(data=> {
    this.loading = false;
      console.log("localhost:5000/fraud/predict says" + JSON.stringify(data));
      this.dataSource = data;
    },
      (error) => {                              //Error callback
        console.error(error)
        this.loading = false;
      });
    console.debug("onSubmit called................."+ this.transaction);
  }

    showFormFn(){
      this.showForm = true;
      this.showJSON = false;
      this.dataSource = [];
    }
    showJSONFn() {
      this.showForm = false;
      this.showJSON = true;
      this.dataSource = [];
    }

}
export interface FraudPrevention {
  type: string;
  amount: number;
  fromAccNo: string;
  destAccNo: string;
  oldbalanceOrg: number;
  newbalanceOrg: number;
  decisionTreeAccuracy: number;
  decisionTreePrediction: string,
  prediction: string,
  indication: string
}
