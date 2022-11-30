import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  cedentials={
    password:'',
    username:''
  }
  constructor() { }

  ngOnInit(): void {
  }
   
  onSubmit(){
    console.log("form is submited",this.cedentials);

    if((this.cedentials.password!="" || this.cedentials.password!=null || this.cedentials.password!=undefined)
     && (this.cedentials.username!="" || this.cedentials.username!=null || this.cedentials.username!=undefined)){
      console.log("need to submit form");
    }else{
      
    }
  }

}
