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

  transaction={
    type:'',
    amount:0.0,
    fromAccNo:'',
    destAccNo:'',
    oldbalanceOrg:0.0
  }

  constructor(private http: HttpClient, private service : RestApiService) { }

  ngOnInit(): void {
  }

  onSubmit(){

    this.transaction.type =  "TRANSFER";
    this.transaction.amount =  8000.00;
    this.transaction.fromAccNo =  "C905679615";
    this.transaction.destAccNo=  "CC572787442";
    this.transaction.oldbalanceOrg =  10000.00;


    this.http.post<Transaction>("http://localhost:5000/fraud/predict", this.transaction).subscribe(data=> {
      console.log("localhost:5000/fraud/predict says" + JSON.stringify(data));
    });
    //this.service.createTransaction(this.transaction);




    console.debug("onSubmit called................."+ this.transaction);
  }



}
